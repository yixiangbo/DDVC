"""
1217 选择测试集的所有数据进行测试，取平均值
1229 按照论文用康达进行测试  batch值应为1
"""
import argparse
import sys
import re
import numpy as np
sys.path.append('image_comp')
import torch
import time
from torch.autograd import Variable
import configparser
import os
import math
from metric import *
import imageio
import cv2
import torch.utils.data as data
import datasetDistribute0318 as datasetDistribute
import dataset_decoder 
import networkDistribute_pyramid_big as networkDistribute
from torchvision import transforms
import analysis_lstm as side_net
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#gpu_num = torch.cuda.device_count()
def file_name(path):
    file_list=[]
    dir = os.listdir(path)
    for name in dir:
        for root, dirs, files in os.walk(os.getcwd()):
            for tt in range(len(files)):
                file_list.append(files[tt]) #当前路径下所有非目录子文件
    return file_list
def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
def get_args(filename,section):
    args = {}
    config = configparser.RawConfigParser()
    config.read(filename)
    options = config.options(section)
    print(len(options))
    for t in range(len(options)):
        if config.get(section,options[t] ).isdigit():
            args[options[t]] = int(config.get(section,options[t] ))
        else:
            try:
                float(config.get(section,options[t] ))
                args[options[t]] = float(config.get(section,options[t] ))
            except:
                args[options[t]] = config.get(section, options[t])
    return args

def init_lstm_test(name,batch_size, height, width):
    """
    1204 添加  用函数替换
    """
    if name == 'encoder':
        with torch.no_grad():
            encoder_h_1 = (Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)))
            encoder_h_2 = (Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)))
            encoder_h_3 = (Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)))

            decoder_h_1 = (Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)))
            decoder_h_2 = (Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)))
            decoder_h_3 = (Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)))
            decoder_h_4 = (Variable(torch.zeros(batch_size, 64*2, height // 2, width // 2)),
                           Variable(torch.zeros(batch_size, 64*2, height // 2, width // 2)))

            encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
            encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
            encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

        return (encoder_h_1, encoder_h_2, encoder_h_3,
                decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
    if name == 'decoder':
        with torch.no_grad():    

            decoder_h_1 = (Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)),
                           Variable(torch.zeros(batch_size, 256*2, height // 16, width // 16)))
            decoder_h_2 = (Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)),
                           Variable(torch.zeros(batch_size, 256*2, height // 8, width // 8)))
            decoder_h_3 = (Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)),
                           Variable(torch.zeros(batch_size, 128*2, height // 4, width // 4)))
            decoder_h_4 = (Variable(torch.zeros(batch_size, 64*2, height // 2, width // 2)),
                           Variable(torch.zeros(batch_size, 64*2, height // 2, width // 2)))


            decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
            decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
            decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
            decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

        return (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
def psnr01(img1, img2):
    #mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def load_model(model):
    encoder.load_state_dict(torch.load(model,map_location=torch.device('cpu')))
    binarizer.load_state_dict(torch.load(model.replace('encoder', 'binarizer'),map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder'),map_location=torch.device('cpu')))
    return encoder,binarizer,decoder
def load_model_side(model, f):
    model.load_state_dict(torch.load(f,map_location=torch.device('cpu')))
def load_set(test,batch_size):
    test_set = datasetDistribute.ImageFolder(is_train=False, root=test)
    # 1210 加载test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    return test_loader
def flag_judge(flag,imgPre,imgMid,imgNext):
    if flag == 1:  # 前
        dataSide = imgPre
    elif flag == 2:  # 中
        dataSide = imgMid
    elif flag == 3:  # 后
        dataSide = imgNext
    elif flag == 4:  # 噪声
        dataSide = torch.FloatTensor(gasuss_noise(imgMid.numpy()))
    elif flag == 5:  # 前一帧和后一帧
        dataSide = x = torch.cat([imgPre, imgNext], dim=1)
    elif flag == 6:
        dataSide = (imgPre + imgNext) / 2
    return dataSide

def encoder_img(train_dataset,max_batch,encoder,binarizer,flag,level,iterations,output_name):
    if os.path.exists('wzall.npz'):
        os.system('rm wzall.npz')
        print('delete old wzall.npz !')

    test_set = datasetDistribute.ImageFolder(is_train=False, root=train_dataset)
    # 1210 加载test
    test_loader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    file = []
    for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(test_loader):
        if batch >=149:
            break;
        if batch%2 ==0:
            continue
        #print('filename: ',filename,filenamePre, filenameNext)
        encoder.eval()
        binarizer.eval()
        # decoder.eval()

        imgPre = imgAll[:, 0:1, :, :]
        imgMid = imgAll[:, 1:2, :, :]
        imgNext = imgAll[:, 2:3, :, :]
        data1 = imgMid

        #dataSide = flag_judge(flag,imgPre,imgMid,imgNext)
        dataSide = (imgPre + imgNext) /2
        #dataSide = torch.cat((imgPre,imgNext), dim=1)
        with torch.no_grad():
            image = Variable(data1.cuda())
        dataSide = Variable(dataSide.cuda())

        res = image - 0.5
        dataSide = dataSide - 0.5

        (encoder_h_1, encoder_h_2, encoder_h_3,
         decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm_test(
            name = 'encoder',batch_size=image.size(0), height=image.size(2), width=image.size(3))
        codes = []
        for iters in range(iterations):
            # Encode.
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            # Binarize.
            code = binarizer(encoded)
           # print(code.shape)
            # Decode.
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,dataSide)

            res = res - output  # Variable

            codes.append(code.data.cpu().numpy())
        output = output+0.5 
        #print(psnr01(imgMid.cpu().detach().numpy(), output.cpu().detach().numpy()))   
        # codes = (np.stack(codes).astype(np.int8) + 1) // 2
        #
        # export = np.packbits(codes.reshape(-1))
        file.append(codes)
        # np.savez_compressed(output_name+str(batch), shape=codes.shape, codes=export)
        #print(batch)
    if os.path.exists('wzall.npz'):
        os.system('rm wzall.npz')
        print("delete last wzall.npz !")
    codes = (np.stack(file).astype(np.int8) + 1) // 2
    export = np.packbits(codes.reshape(-1))
    np.savez_compressed(output_name+'all', shape=codes.shape, codes=export)
def decoder_img(input_file,input_name,tframe,kframe,max_batch,encoder,binarizer,flag,level,iterations,output_file,output_name):
    if os.listdir(output_file):
        os.system('rm {}/*'.format(output_file))
        print('Decoder WZ frames, WZ frames folder clear !!')
    
    test_set = dataset_decoder.ImageFolder(is_train=False, root=kframe)
    # 1210 加载test
    kframe_loader = data.DataLoader(
        dataset=test_set, batch_size=1, shuffle=False, num_workers=0)
    
    test_set0 = datasetDistribute.ImageFolder(is_train=False, root=tframe)
        # 1210 加载test
    #tframe_loader = data.DataLoader(
    #    dataset=test_set0, batch_size=1, shuffle=False, num_workers=1)
    num = []
    num1 = []
    ssim=[]
    content = np.load(input_file + 'all' + '.npz')
    codes_all = np.unpackbits(content['codes'])
    codes_all = np.reshape(codes_all, content['shape']).astype(np.float32) * 2 - 1

    codes_all = torch.from_numpy(codes_all)
    r = 1
    for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(kframe_loader):
        #print('encoder:output_img'+'.npz')
        #print(filename, filenamePre, filenameNext)
        #print(batch)
        codes = codes_all[batch]
        iters, batch_size1, channels, height, width = codes.size()
        height = height * 16
        width = width * 16
        with torch.no_grad():
            codes = Variable(codes)
        codes = codes.cuda()
        # decoder.eval()
        (imgAll0, filename0, filenamePre0, filenameNext0) = test_set0[r]
        r = r+2
        #print(filename0)
        imgMid0 = imgAll0[ 1:2, :, :]
        imgPre0 = imgAll0[ 0:1, :, :]
        imgNext0 = imgAll0[ 2:3, :, :]


        imgPre = imgAll[:, 0:1, :, :]
        imgMid = imgAll[:, 1:2, :, :]
        imgNext = imgAll[:, 2:3, :, :]
        #data = imgMid
        #print('a:',psnr01(imgMid0.cpu().detach().numpy(), imgMid.cpu().detach().numpy()))
        #dataSide = imgPre
        #dataSide = flag_judge(flag,imgPre,imgMid,imgNext)
        dataSide = ((imgPre + imgMid) /2)
        #.unsqueeze(0)
        #dataSide = torch.cat((imgPre,imgNext), dim=1)
        dataSide = Variable(dataSide.cuda())    
        #dataSide = Variable(dataSide)
        dataSide = dataSide - 0.5

        (decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm_test(name = 'decoder',
                batch_size=batch_size1, height=height, width=width)
        #print('imgPre.shape',imgPre.shape)
        #print('code.shape',codes.shape)
       
        image = torch.zeros(imgPre.shape) + 0.5  # 确定输出的尺寸 1204添加
        image = Variable(image)  # 转化成相同的数据类型 1204
        #print(psnr01(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        #print('length:',len(codes))
        for iters in range(min(iterations, codes.size(0))):
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,dataSide)
            image = image + output.data.cpu()
            #print('iters',iters)
        #image = image -0.5
        #print('write: '+output_file+'/'+output_name+str(r-1)+'.png')
        #print(psnr01(imgMid0.cpu().detach().numpy(), image.cpu().detach().numpy()))
        num1.append(psnr01(imgMid0.cpu().detach().numpy(), image.cpu().detach().numpy()))
        
        
        
        side_h_1 = (Variable(torch.zeros(image.size(0), 128//2, image.size(2) , image.size(3) )),
                       Variable(torch.zeros(image.size(0), 128//2, image.size(2) , image.size(3) )))
        side_h_2 = (Variable(torch.zeros(image.size(0), 256//2, image.size(2) , image.size(3) )),
                       Variable(torch.zeros(image.size(0), 256//2, image.size(2) , image.size(3) )))
        side_h_3 = (Variable(torch.zeros(image.size(0), 128//2, image.size(2) , image.size(3) )),
                       Variable(torch.zeros(image.size(0), 128//2, image.size(2) , image.size(3) )))
        side_h_4 = (Variable(torch.zeros(image.size(0), 64//2, image.size(2) , image.size(3) )),
                       Variable(torch.zeros(image.size(0), 64//2, image.size(2) , image.size(3) )))
        
        for iteration in range(1):
            input_img = torch.cat((imgPre, image.cpu(),imgMid), dim=1)
            recon_image, side_h_1, side_h_2, side_h_3,side_h_4 = net_side(
                input_img, side_h_1, side_h_2, side_h_3,side_h_4)
            image = recon_image
            #print('PSNR:',msssim(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        
        
        
        #input = torch.cat((imgPre, image.cpu(),imgMid), dim=1)   
        #recon_image = net_side(input)
        
        #print('PSNR:',msssim(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        #print('####',psnr01(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        num.append(psnr01(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        #ssim.append(msssim(imgMid0.cpu().detach().numpy(), recon_image.cpu().detach().numpy()))
        imageio.imwrite(
            os.path.join(output_file+'/'+output_name+'{:02d}.png'.format(r-1)),
            np.squeeze(recon_image.cpu().detach().numpy().clip(0, 1) * 255.0).astype(np.uint8))
    #print("Decoder WZ frames finish , Average PSNR：",np.mean(num1),np.mean(num))
    #print("Decoder WZ frames finish , Average MSSSIM：",np.mean(ssim))
    #psnr,bit = pipei_stat('stats.dat')
    #print('video all average psnr :',( np.mean(num)+psnr)/2.0)
    print("Decoder WZ frames finish , Average PSNR：",np.mean(num1),np.mean(num))
    print('video all average psnr :',( np.mean(num)+pipei('log.txt'))/2.0)
def yuv2img(file_name,save_path, height, width, start_frame):
    """
    :param file_name: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param start_frame: 起始帧
    :return: None
    """
    #print('sdfsdf')
    if  os.listdir(save_path):
        os.system('rm {}/*'.format(save_path))
        print('origin image frames folder clear !!')
    
    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置
    print(fp_end//300)
    frame_size = height * width #* 3 // 2  # 一帧图像所含的像素个数
    num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This {} file has {} frame imgs!".format(file_name,num_frame))
    fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    #print("Extract imgs start frame is {}!".format(start_frame + 1))

    for i in range(num_frame - start_frame):
        yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode

        img = yyyy_uv.reshape((height * 2 // 2, width)).astype('uint8')  # NV12 的存储格式为：YYYY UV 分布在两个平面（其在内存中为 1 维）
        # bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_I420)  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2GRAY_I420)
        # bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(save_path+'/hall_qcif_%05d.png' % (i + 1), img)  # 改变后缀即可实现不同格式图片的保存(jpg/bmp/png...)
    print("Extract frame {}".format(i + 1))

    fp.close()
    print("{} convert to image finish!".format(file_name))
    return None
def encode_key(filename,QP):
    if os.path.exists('str.bin'):
        os.system('rm str.bin')
        os.system('rm rec.yuv')
        print('delete old str.bin and rec.yuv !')
    
    if os.path.exists('str.bin'):
        print('error')

    if os.path.exists('log.txt'):
        os.system('rm log.txt')
        print('delete old log.txt !')
    print('encodering Key frames ......')
    commd = '/home/whut4/yixiangbo/HM/HM-16.20+SCM-8.8/bin/TAppEncoderStatic -c key_cfg/encoder_intra_main_rext.cfg -c key_cfg/BQSquare.cfg  -i {} -ts 2 -f 149 -q {} > log.txt'.format(filename,QP)
    print('asd:',os.getcwd())
    #../bin/TAppEncoderStatic -c ../cfg/encoder_intra_main_rext.cfg -c ../cfg/per-sequence/BQSquare.cfg  -i coastguard_black.yuv -ts 2 -f 149 -q 32
    if os.system(commd)!=0:
        print('encodering Key frames fail !')


    print("Key frames encoder finish !")
def pipei(dir):
    while not os.path.exists(dir):  # 判断文件是否存在
        dir = input('Cann\'t find the file,Please input the correct file dir:')
    data = open(dir, 'r')  # 打开文件
    flag = 0
    p = re.compile(r'SUMMARY')
    for lines in data:
        value = lines.split('\t')  # 读出每行
        # print(value)
        if flag:
            if flag ==1:
                # print(str(value)[-13:-5])
                psnr = float(str(value)[-13:-5])
                print(psnr)
            flag =flag -1
        if re.search(p, str(value)):
            flag = 2
    data.close()
    return psnr
def decode_key(de_key_path,height, width, start_frame):
    if  os.listdir(de_key_path):
        os.system('rm {}/*'.format(de_key_path))
        print('Key frames folder clear !!')

    if os.path.exists('rec.yuv'):
        #yuv2img('rec.yuv', de_key_path, 144, 176, 0)
        fp = open('rec.yuv', 'rb')
        fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
        fp_end = fp.tell()  # 获取文件尾指针位置
        frame_size = height * width  # * 3 // 2  # 一帧图像所含的像素个数
        num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
        fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
        t = 1
        for i in range(num_frame - start_frame):
            yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
            for j in range(frame_size):
                yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode

            img = yyyy_uv.reshape((height * 2 // 2, width)).astype('uint8')
            cv2.imwrite(de_key_path + '/hall_qcif_%05d.png' % (t), img)
            t = t+2
        fp.close()

        print('key frames decode to {} success !'.format(de_key_path))
        print('key frames psnr :',pipei('log.txt'))
    else:
        print("rec.yuv not exist !")
def video_cat(path1,path2,num=150):
    if os.path.exists('result.yuv'):
        os.system('rm result.yuv')
        print('delete result.yuv,and create new result.yuv !')
    fp = open('result.yuv', 'wb+')
    ssim = []
    for i in range(1,num):
        if i%2!=0:
            ssim.append(msssim('/home/whut4/yixiangbo/test_video_comp_HM/img_path/hall_qcif_%05d.png'%(i),path1+ '/hall_qcif_%05d.png' % (i)))
            image = Image.open(path1+ '/hall_qcif_%05d.png' % (i)).convert('L')
        else:
            ssim.append(msssim('/home/whut4/yixiangbo/test_video_comp_HM/img_path/hall_qcif_%05d.png'%(i),path2+'/result_img{:02d}.png'.format(i)))
            image = Image.open(path2+'/result_img{:02d}.png'.format(i)).convert('L')
        image = np.asarray(image)
        fp.write(image)
    fp.close()
    print('video image SS-SSIM average: ',np.mean(ssim))
    print('Images(WZ and Key frames) merge into result.yuv success !')






# torch.load 加载模型
# replace 替换字符串
#encoder.load_state_dict(torch.load(args['model']))
#binarizer.load_state_dict(torch.load(args['model'].replace('encoder', 'binarizer')))
#decoder.load_state_dict(torch.load(args['model'].replace('encoder', 'decoder')))
def get_bps(path1,path2):
    key_size = os.path.getsize(path1)*8.0
    wz_size = os.path.getsize(path2)*8.0
    bps = (key_size/75.0*7.5+wz_size/74.0*7.5)/1024.0
    return bps
if(sys.argv[1] == 'encoder'):
    args = get_args("config.ini",'encoder')
    
    yuv2img(args['filename'],args['img_path'],144, 176,0)
    start = time.clock()
    encode_key(args['filename'],args['key_qp'])
    
    encoder = networkDistribute.EncoderCell().cuda()
    binarizer = networkDistribute.Binarizer().cuda()
    decoder = networkDistribute.DecoderCell().cuda()

    model_side = side_net.SideCell()
    load_model(args['model'])
    print('Encodering wz frames......')
    encoder_img(args['test'],args['max_batch'],encoder,binarizer,args['flag'],args['level'],args['iterations'],args['output_name'])
    elapsed = (time.clock() - start)
    print("time:",elapsed)
    print('WZ frames encoder finish, file wzall.npz !')
    
    print("encoder bitrate : {} kbps".format(get_bps('str.bin','wzall.npz')))
elif (sys.argv[1] == 'decoder'):
    start = time.clock()
    encoder = networkDistribute.EncoderCell().cuda()
    binarizer = networkDistribute.Binarizer().cuda()
    decoder = networkDistribute.DecoderCell().cuda()

    model_side = side_net.SideCell()
    args = get_args("config.ini",'decoder')
    load_model(args['model'])
    global_step = load_model_side(model_side, args['sidetrain'])
    net_side = model_side.cuda()
    #net_side = torch.nn.DataParallel(net_side, list(range(gpu_num)))
    net_side = torch.nn.DataParallel(net_side)
    decode_key(args['de_key_path'], 144, 176,0)
    print('Decoder WZ start......')
    decoder_img(args['input_file'],'output_img',args['test0'],args['de_key_path'],args['max_batch'],encoder,binarizer,args['flag'],args['level'],args['iterations'],args['output_file'],args['output_name'])
    video_cat(args['de_key_path'],args['output_file'],num=150)
    elapsed = (time.clock() - start)
    print("time:",elapsed)
else:
    print(sys.argv[1])
    print("please input a parameter (encoder or decoder)")
    exit()
