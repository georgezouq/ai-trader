import rqalpha
import os
import numpy as np

from rqalpha.api import *
from sklearn.preprocessing import MinMaxScaler



# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    context.s1 = '600036.XSHG'
    update_universe(context.s1)
    context.has_save_data = False

    mode = 'run'
    market = 'stock'
    training_data_ratio = 0.9
    train_steps = 30000

    base = config.get('base')

    codes = ['600036']
    env = Market(codes, start_date=base.get('start_date'), end_date=base.get('end_date'), **{
        "market": market,
        "use_sequence": True,
        "scaler": MinMaxScaler,
        "mix_index_state": True,
        "training_data_ratio": training_data_ratio
    })

    model_name = 'DualAttnRNN'  # os.path.basename(__file__).split('.')[0]

    context.bar_list_origin = []
    context.bar_list = []
    context.scale = MinMaxScaler()
    context.algorithm = Algorithm(
        tf.Session(config=alg_config), env, env.seq_length, env.data_dim, env.code_count, **{
            "mode": mode,
            "hidden_size": 5,
            "enable_saver": True,
            "train_steps": train_steps,
            "enable_summary_writer": True,
            "save_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "model"),
            "summary_path": os.path.join(CHECKPOINTS_DIR, "SL", model_name, market, "summary"),
        }
    )


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    pass


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    s1 = bar_dict[context.s1]
    price = [s1.open, s1.high, s1.low, s1.close, s1.volume]
    context.bar_list_origin.append(price)

    scale = context.scale.fit(context.bar_list_origin)
    price_scaled = scale.transform([price])
    context.bar_list.append(price_scaled[0])

    # if not enough bar
    if len(context.bar_list) < context.algorithm.seq_length * 2 + 2:
        return

    # frm = len(context.bar_list)-context.algorithm.seq_length
    x1 = context.bar_list[-context.algorithm.seq_length*2:-context.algorithm.seq_length]
    x2 = context.bar_list[-context.algorithm.seq_length:]

    x = [x1, x2]
    _, loss = context.algorithm.predict(x)
    res = np.append(_, [0, 0, 0, 0])
    predict = scale.inverse_transform([res])
    print('predict', predict)
    pass


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass


rqalpha.run_func(init=init,
                 before_trading=before_trading,
                 handle_bar=handle_bar,
                 after_trading=after_trading,
                 config=config)
