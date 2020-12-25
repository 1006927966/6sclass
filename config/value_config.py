##############for dataset############
HEIGHT = 224
WIDTH = 224
#DATAPATH = "/defaultShare/share/wujl/83/classibox/hfdata/croprote/traindata"
DATAPATH = "/VisualGroup/share/wujl/83/ddb_label"
THREAD = 32
BATCHSIZE=128
LOADERNAME="zlc_aware_label"
#############for train ##############
GPUS = '0,1'
LOADING = False 
LOADPATH = ''
MODELNAME = "BCNN"
NUMCLASS = 17
INITLR = 0.01
LOGPATH = '/VisualGroup/share/wujl/83/master_models/resnext101_19_mixcut/log'
WD = 1e-4
WARM = 4
LRTAG = 'MultiStep'
EPOCHS=100
MIXUP = True
Cut = True
############# for save ###############
SAVEFREQ = 20
#MODELSAVEPATH = "/defaultShare/share/wujl/83/hfmodels/v3"
MODELSAVEPATH="/VisualGroup/share/wujl/83/master_models/bcnn_label"
# save frozen
#MODELSAVEPATH = "/defaultShare/share/wujl/83/master_models/resnext101_22/"
############# for eval ###############
MAPLIST = ['badcase', 'bc', 'feigang', 'hz', 'zx', 'laji00', 'laji01', 'laji02', 'laji03',
           'lf', 'pip', 'xiaojian', 'xs', 'zw', 'zhufei', 'people', 'gulixj']
#MAPLIST = ['zq', 'luan']
# MAPLIST = ['badcase', 'bancai_luan', 'bancai_zq', 'feigang', 'hanzha_zq', 'hanzha_l',
#              'zhixiang_luan', 'zhixiang_zhenqgi', 'laji00', 'laji01', 'laji02', 'laji03',
#              'luanfang', 'pip_l', 'pip_mid', 'pip_zq', 'xiaojian_luan', 'xiaojian_zhengqi',
#              'xianshu_luan', 'xianshu_zhengqi', 'zangwu', 'zhufei', 'people', 'gulixj', 'cr']

