import math
import torch
from rl_m19.config import Config
from rl_m19.network import Nematode
from rl_m19.utils import Logger, Plotter
from dqn_agent_ext import DQNAgentExt
from futures_ctp_env import FuturesCTP, BacktestData

config = Config()
config.episode_lifespan = 10**4
config.episodes = 10**4
config.batch_size = 64  # should be power of 2, e.g. 2^n
config.gamma = 0.999
# config.eps_fn = lambda s: 0.9
config.eps_fn = lambda s: 0.05 + (0.90 - 0.05) * math.exp(-1. * s / BacktestData.TRAINED)
config.lr = 0.01
config.replay_size = BacktestData.TRAINED
config.target_update_freq = 5

config.logger = Logger()
config.plotter = Plotter()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.env = FuturesCTP(config.device, config.plotter)
config.state_dim = config.env.state_dim
config.action_dim = config.env.action_dim

config.policy_net = Nematode(config)
config.target_net = Nematode(config)
config.optimizer = torch.optim.RMSprop(config.policy_net.parameters(), config.lr)
config.loss_fn = torch.nn.MSELoss()

if __name__ == '__main__':
    agent = DQNAgentExt(config)
    agent.episodes_learn()
    config.env.render()
    config.env.close()
    config.plotter.plot_end()

'''
SHFE.RB 螺纹钢，1手10吨，保证金比例14%, 价格最小变动单位1
'''
'''
CFFEX.IC    中证500期货主力连续合约   
CFFEX.IF    沪深300期货主力连续合约   
CFFEX.IH    上证50期货主力连续合约    
CFFEX.T     10年期国债期货主力连续合约
CFFEX.TF    5年期国债期货主力连续合约 
CFFEX.TS    2年期国债期货主力连续合约 
CZCE.AP     苹果主力连续合约          
CZCE.CF     棉花主力连续合约          
CZCE.CJ     红枣主力连续合约          
CZCE.CY     棉纱主力连续合约          
CZCE.FG     玻璃主力连续合约          
CZCE.JR     粳稻主力连续合约          
CZCE.LR     晚籼稻主力连续合约        
CZCE.MA     甲醇主力连续合约          
CZCE.OI     菜油主力连续合约          
CZCE.PM     普麦主力连续合约          
CZCE.RI     早籼稻主力连续合约        
CZCE.RM     菜粕主力连续合约          
CZCE.RS     菜籽主力连续合约          
CZCE.SA     纯碱主力连续合约          
CZCE.SF     硅铁主力连续合约          
CZCE.SM     锰硅主力连续合约          
CZCE.SR     白糖主力连续合约          
CZCE.TA     PTA主力连续合约          
CZCE.UR     尿素主力连续合约          
CZCE.WH     强麦主力连续合约          
CZCE.ZC     动力煤主力连续合约        
DCE.A       豆一主力连续合约          
DCE.B       豆二主力连续合约          
DCE.BB      胶合板主力连续合约        
DCE.C       玉米主力连续合约          
DCE.CS      玉米淀粉主力连续合约      
DCE.EB      苯乙烯主力连续合约        
DCE.EG      乙二醇主力连续合约        
DCE.FB      纤维板主力连续合约        
DCE.I       铁矿石主力连续合约        
DCE.J       焦炭主力连续合约          
DCE.JD      鸡蛋主力连续合约          
DCE.JM      焦煤主力连续合约          
DCE.L       塑料主力连续合约          
DCE.M       豆粕主力连续合约          
DCE.P       棕榈油主力连续合约        
DCE.PP      聚丙烯主力连续合约        
DCE.RR      粳米主力连续合约          
DCE.V       PVC主力连续合约          
DCE.Y       豆油主力连续合约          
INE.NR      20号胶主力连续合约          
INE.SC      原油主力连续合约          
SHFE.AG     白银主力连续合约          
SHFE.AL     铝主力连续合约            
SHFE.AU     黄金主力连续合约          
SHFE.BU     沥青主力连续合约          
SHFE.CU     铜主力连续合约            
SHFE.FU     燃油主力连续合约          
SHFE.HC     热轧卷板主力连续合约      
SHFE.NI     镍主力连续合约            
SHFE.PB     铅主力连续合约            
SHFE.RB     螺纹钢主力连续合约        
SHFE.RU     橡胶主力连续合约          
SHFE.SN     锡主力连续合约            
SHFE.SP     纸浆主力连续合约          
SHFE.SS     不锈钢主力连续合约        
SHFE.WR     线材主力连续合约          
SHFE.ZN     锌主力连续合约
'''
