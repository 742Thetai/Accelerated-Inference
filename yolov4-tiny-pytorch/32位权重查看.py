import struct

# 读取二进制文件中的权重参数
def read_weights_from_bin(file_path):
    weights = []
    with open(file_path, 'rb') as f:
        while True:
            # 从文件中读取4字节（32位）的浮点数
            weight_bytes = f.read(4)
            if not weight_bytes:
                break
            # 将字节转换为32位浮点数
            weight = struct.unpack('f', weight_bytes)[0]
            weights.append(weight)
    return weights

# 将权重参数保存到文本文件中
def save_weights_to_txt(weights, output_file):
    with open(output_file, 'w') as f:
        for weight in weights:
            # 将权重参数写入文本文件，保留小数点后32位
            f.write('{:.32f}\n'.format(weight))

# 示例用法
if __name__ == "__main__":
    # 读取二进制文件中的权重参数
    weights = read_weights_from_bin('/home/wma/yolov4-tiny-pytorch-master/HLS_pj/folded_weight/BasicConv1/b.bin')
    
    # 将权重参数保存到文本文件中
    save_weights_to_txt(weights, 'weights.txt')