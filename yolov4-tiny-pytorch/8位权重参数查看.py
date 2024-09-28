

import struct
with open('/home/wma/yolov4-tiny-pytorch-master/HLS_pj/folded_weight/BasicConv1/b.bin', 'rb') as f:
    content = f.read()

# 解析int8整数
parameters = []
for byte in content:
    # 使用struct.unpack将每个字节解析为有符号的8位整数
    int_val = struct.unpack('b', bytes([byte]))[0]
    parameters.append(int_val)

print(parameters)


