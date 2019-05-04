import numpy
from settings import CHARACTER, CAPTCHA_NUMBER


def one_hot_encode(value: list) -> tuple:
    """编码，将字符转为独热码
    vector为独热码，order用于解码
    """
    order = []
    shape = CAPTCHA_NUMBER * len(CHARACTER)
    vector = numpy.zeros(shape, dtype=float)
    for k, v in enumerate(value):
        index = k * len(CHARACTER) + CHARACTER.get(v)
        vector[index] = 1.0
        order.append(index)
    return vector, order


def one_hot_decode(value: list) -> str:
    """解码，将独热码转为字符
    """
    res = []
    for ik, iv in enumerate(value):
        val = iv - ik * len(CHARACTER) if ik else iv
        for k, v in CHARACTER.items():
            if val == int(v):
                res.append(k)
                break
    return "".join(res)


if __name__ == '__main__':
    code = '0A2JYD'
    vec, orders = one_hot_encode(code)
    print('将%s进行特征数字化处理' % code)
    print('特征数字化结果：%s' % vec)
    print('字符位置：%s' % orders)
    print('根据特征数字化时的字符位置进行解码，解码结果为：%s' % one_hot_decode(orders))

