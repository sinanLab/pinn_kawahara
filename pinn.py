import lightning as pl
from pathlib import Path
import torch, yaml, h5py, warnings, gc
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger     # Comet Logger, Neptune Logger, TensorBoard Logger can be used with PL
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import EarlyStopping, BatchSizeFinder
from lightning.pytorch import Trainer

'''
This work is mainly adopted form the lightning.ai (formerly Pytorch_lightning) [6] 
and taken help from ChatGPT multiple times
'''

gc.set_threshold(0)

def load_config(file):
    with open(file, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

config = load_config('config.yml')

warnings.filterwarnings("ignore")    
torch.set_float32_matmul_precision('high')

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: Path = config['data']['dataset_dir'], 
                 train_size: float = config['data']['train_size'], 
                 batch_size: int = config['train']['batchsize'], 
                 num_workers: int = config['train']['num_workers'],
                 ):
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.persistance = False
        if num_workers > 0:
            self.persistance = True
        self.batch_size = batch_size
        self.train_size = train_size
        self.train_dataset = None  # Initialize train_dataset attribute
        self.val_dataset = None    # Initialize val_dataset attribute
        self.test_dataset = None   # Initialize test_dataset attribute

    def prepare_data(self):
        # Open the HDF5 file
        with h5py.File(self.data_dir, 'r') as f:
            input_variables = torch.Tensor(f['input_variables'][:])
            target_solution = torch.Tensor(f['target_solution'][:])
            
        # Calculate mean and standard deviation
        input_mean = input_variables.mean(dim=0)
        input_std = input_variables.std(dim=0)

        # Apply standardization
        input_variables = (input_variables - input_mean) / input_std

        # Create the TensorDataset using the loaded data
        self.dataset = torch.utils.data.TensorDataset(input_variables, target_solution)

    def setup(self):
        train_size = int(self.train_size * len(self.dataset))
        test_size = len(self.dataset) - train_size

        # Create the generator
        generator = torch.Generator(device=config['train']['rank'])
        generator.manual_seed(42)
            
        self.train_dataset, self.val_dataset= torch.utils.data.random_split(
           self.dataset, 
           [train_size, test_size], 
           generator = generator # ChatGPT
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                          persistent_workers = self.persistance,
                          drop_last = True,
                          pin_memory = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                          persistent_workers = self.persistance,    # [3]
                          pin_memory = True,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(self.test_dataset, 
    #                       persistent_workers = self.persistance,
    #                       pin_memory = True,
    #                       batch_size=self.batch_size, 
    #                       num_workers=self.num_workers)

class NN(torch.nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int):
        super(NN, self).__init__()
        
        self.act1 = torch.nn.Tanh()
        # self.l1 = torch.nn.Linear(input_size, 64)
        # self.dropout = torch.nn.Dropout(p=0.5)
        # self.conv1d = torch.nn.Conv1d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        # # define batch normaliation layer
        # self.bn1 = torch.nn.BatchNorm1d(num_features=16)
        # self.act2 = torch.nn.Tanh()
        # self.bilstm = torch.nn.LSTM(input_size=1, hidden_size=64, num_layers=5, batch_first=True, bidirectional= True)
        # # define batch normaliation layer
        # self.bn1 = torch.nn.BatchNorm1d(num_features=16)
        # self.l2 = torch.nn.Linear(128, 64)
        # self.l3 = torch.nn.Linear(64, output_size)
        # self.act3 = torch.nn.Tanh()
        
        # simple model
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, y):
        # x = self.l1(y)
        # x = self.act1(x)
        # x = x.reshape(-1, 1, 64)
        # x = x.transpose(1, 2)
        # x = self.conv1d(x)
        # x = self.bn1(x)
        # x = self.dropout(x)
        # x = self.act2(x)
        # x, (hn, cn) = self.bilstm(x)
        # x = self.bn1(x)
        # x = self.dropout(x)
        # x = self.l2(x)
        # x = self.act3(x)
        # x = self.l3(x)
        # x = x[:, 0, :]
        # x = x.view(-1, 1)
        
        # simple model
        x = self.linear1(y)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = self.linear3(x)
        x = self.act1(x)
        return x

class Kawahara(pl.LightningModule):
    def __init__(self, Model):
        super().__init__()
        
        self.R2_score = R2Score()                           # R2 score metric
        self.mae_metric = MeanAbsoluteError()               # Mean Absolute Error metric
        self.mse_metric = MeanSquaredError()                # Mean Squared Error metric
        
        self.a_1 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)  # Require gradients for nu
        self.a_2 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)  # Require gradients for m
        self.a_3 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)  # Require gradients for rho
        
        # call models
        self.model = Model
        
    def forward(self, inputs):
        return self.model(inputs)

    def pinn_loss(self, tx, target):
        tx.requires_grad = True

        with torch.backends.cudnn.flags(enabled=False):
            pred_u = self.forward(tx)  # Run the forward pass of your RNN
        
        du_dtx = torch.autograd.grad(outputs=pred_u, inputs=tx, 
                                     grad_outputs=torch.ones_like(pred_u), 
                                     allow_unused=True, retain_graph=True)[0]
        du_dt = du_dtx[:, 0]
        du_dx = du_dtx[:, 1]

        # Calculate the second-order derivatives
        for _ in range(2):   # here 2 means that the second order derivative
            u_xxx = torch.autograd.grad(outputs=pred_u, inputs=tx, grad_outputs=torch.ones_like(pred_u), allow_unused=True, create_graph=True)[0]
            u_xxx = u_xxx[:, 1]
            
        for _ in range(3):   # here  means that the second order derivative
            u_xxxxx = torch.autograd.grad(outputs=pred_u, inputs=tx, grad_outputs=torch.ones_like(pred_u), allow_unused=True, create_graph=True)[0]
            u_xxxxx = u_xxxxx[:, 1]
            
        def g(x: torch.Tensor) -> torch.Tensor:
            return (105.0 / 169.0) * torch.pow(1 / torch.cosh(x / (2 * torch.sqrt(torch.tensor(13.0)))), 4)
        
        zeros_init = torch.zeros_like(tx[:, 0])
        init = pred_u[:, 0]-g(tx[:, 0]) # u(x,0) - g(x) (initial condition, as u(x, 0) = g(x), then the function g(x) bring to the left hand side)
        init_res = self.mse_metric(init, zeros_init)
        
        # dudx = du_dx.reshape(-1,)
        
        f = du_dt + self.a_1*pred_u*du_dx+self.a_2*u_xxx-self.a_3*u_xxxxx  # [7]
        f = f.mean()
        zeroTarget = torch.zeros_like(f)
        
        res_MSE = self.mse_metric(f,zeroTarget) 
        res_MAE = self.mae_metric(f,zeroTarget) 
        MSE = self.mse_metric(pred_u, target)
        MAE = self.mae_metric(pred_u, target)
        R2_sc = self.R2_score(pred_u, target)
        return f, init_res, res_MSE, res_MAE, MSE, MAE, R2_sc
        
    def update(self, input_vars, y):
            return self.pinn_loss(tx=input_vars, target=y)
    
    def compute_residual(self, batch, y):
        return self.update(batch, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        x, y = x.to(device), y.to(device)
        
        f, init_res, res_MSE, res_MAE, MSE, MAE, R2_sc = self.compute_residual(x, y)
        total_loss = f * config['train']['f_loss_weight'] \
                    + res_MSE * config['train']['xy_loss_weight'] \
                    + res_MAE * config['train']['xy_loss_weight'] \
                    + init_res * config['train']['ic_loss_weight'] \
                    + MSE * config['train']['data_loss_weight'] \
                    + MAE * config['train']['data_loss_weight']
        
        self.log_dict({
            'train_MAE': MAE, 
            'train_MSE': MSE, 
            'train_R2_score': R2_sc,
            'train_total_loss': total_loss,
            'train_residual_MSE': res_MSE, 
            'train_residual_MAE': res_MAE, 
            'train_init_res': init_res, 
            'train_f': f,
            },
            on_step=False,      # on_step means that to show detail of every epoch separately if it's ON. 
            on_epoch=True, 
            prog_bar=True, 
            logger=True)
        self.logger.experiment.add_histogram("Training: MAE", MAE, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: MSE", MSE, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: R2_score", R2_sc, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: total_loss", total_loss, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: res_MSE", res_MSE, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: res_MAE", res_MAE, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: init_res", init_res, global_step=trainer.global_step)
        self.logger.experiment.add_histogram("Training: f", f, global_step=trainer.global_step)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        val_pred_u = self.forward(x)
        val_loss = self.mse_metric(val_pred_u, y)
        self.logger.experiment.add_histogram("Validation: loss", val_loss, global_step=trainer.global_step)
        self.log('val-loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)   
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        val_pred_u = self.forward(x)
        test_loss = self.mse_metric(val_pred_u, y)
        self.log('test-loss', self.test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.logger.experiment.add_histogram("Testing: loss", test_loss, global_step=trainer.global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        device = x.device
        x, y = x.to(device), y.to(device)
        output_u = self.forward(x)
        return output_u
    
    def _common_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        # Forward pass to get model predictions
        output_u = self.forward(x)
        # Compute the loss using the provided metric
        loss = self.mse_metric(output_u, y)
        return loss
    
    def visualization(self):
        return
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=config['train']['base_lr'])
    
class FineTuneBatchSizeFinder(BatchSizeFinder):     # [5]
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)
    
if __name__ == '__main__':
    custom_profiler = PyTorchProfiler(
        on_trace_ready = torch.profiler.tensorboard_trace_handler('tb_logs/Kawahara_model'),
        schedule = torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=20)
    )
    
    logger = TensorBoardLogger('tb_logs', name="Kawahara_model")
    
    # Initialize a PyTorch SummaryWriter
    writer = SummaryWriter(logger.log_dir)
    strategy = DDPStrategy(static_graph=True)   # [4]
    trainer = Trainer(
                    accelerator=config['train']['accelerator'], 
                    # strategy=strategy,
                    devices = config['train']['device'], 
                    max_epochs=config['train']['max_epochs'], 
                    precision=config['train']['precision'], 
                    enable_checkpointing=True,
                    check_val_every_n_epoch=1,
                    detect_anomaly=True,
                    gradient_clip_val=0.5,
                    gradient_clip_algorithm="value",
                    enable_progress_bar=True,
                    enable_model_summary=True, # summary: will show every detail of training after finishing of epochs, such as time etc 
                    profiler=custom_profiler,
                    deterministic=True,
                    logger=logger,
                    # callbacks= [EarlyStopping(monitor="val-loss", mode='min'), 
                                # # FineTuneBatchSizeFinder(milestones=(5, 10)),      # finding the best batch_size to support by the cuda memory,  [5]
                                # ], 
                    )
    
    with trainer.init_module():    # [3]
        datamodule = DataModule()
        datamodule.prepare_data()
        datamodule.setup()
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        # test_loader = datamodule.test_dataloader()
        pytorch_model = NN(
                    input_size=config['model']['in_channel'], 
                    hidden_size=config['model']['num_layers'], 
                    output_size=config['model']['out_channel'], 
                )
        kawahara_problem = Kawahara(pytorch_model)
    
    """
    to train multiple models, [1] can be used
    
    """
    
    trainer.fit(model=kawahara_problem, train_dataloaders=train_loader)
    trainer.validate(model=kawahara_problem, dataloaders=val_loader)
    
    # Iterate over the model parameters and add histograms to TensorBoard
    for name, param in pytorch_model.named_parameters():
        writer.add_histogram(name, param, global_step=trainer.global_step)
    # Close the SummaryWriter
    writer.close()


""" References
[1] https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
[2] https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
[3] https://github.com/Lightning-AI/pytorch-lightning/issues/19373
[4] https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html#ddpstrategy
[5] https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BatchSizeFinder.html#lightning.pytorch.callbacks.BatchSizeFinder
[6] https://lightning.ai/blog/pl-tutorial-and-overview/
[7] Sinan, Muhammad, et al. "On Semianalytical Study of Fractional-Order Kawahara Partial Differential Equation with the Homotopy 
    Perturbation Method." Journal of Mathematics 2021 (2021): 1-11.
"""