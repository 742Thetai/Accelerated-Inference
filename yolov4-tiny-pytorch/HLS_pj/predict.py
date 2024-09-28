from PIL import Image
from yolo import YOLO



if __name__ == "__main__":
    yolo = YOLO()
    # 0: 图片预处理,生成可以直接作为加速器输入的img.bin文件; 1: 根据加速器的输出进行解码、显示
    #mode=0
    mode=int(input('Input mode(0:preprocess, 1:detect):'))
    if mode==0:
        #dir   = input('Output directory:')
        #image = input("Input image:")
        dir      = "yolo_test"  # 生成 img.bin 保存路径
        image    = "/home/wma/yolov4-tiny-pytorch-master/HLS_pj/accelerator_python/yolo_test/0065.jpg"# 测试图片路径
        try:
            image = Image.open(image)
        except:
            print('Open Error! Try again!')
        yolo.preprocess_image(image,dir)
        print("Finish image preprocess!")
    elif mode==1:
        #dir      = input('Input directory:')
        #img_name = input("Input image name:")
        dir       = "/home/wma/yolov4-tiny-pytorch-master/HLS_pj/yolo_test"
        img_name  = "img.png"
        result_path = dir + "result.png"
        try:
            image = Image.open("/home/wma/yolov4-tiny-pytorch-master/HLS_pj/accelerator_python/yolo_test/img.png") # 打开测试所用图片
        except:
            print('Open Error! Try again!')
        else:
            r_image = yolo.detect_image(image, dir) # 根据 out0, out1 在原图上绘制锚框
            # r_image.show()
            r_image.save(result_path)

    
