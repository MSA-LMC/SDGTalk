#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import io
import os
import random
from warnings import warn

import lmdb
import torch
import torchvision
import numpy as np

class LMDBEngine:
    def __init__(self, lmdb_path, write=False):
        """
        lmdb_path: str, path to the lmdb database.
        write: bool, whether to open the database in write mode.
        """
        self._write = write
        self._manual_close = False
        self._lmdb_path = lmdb_path
        if write and not os.path.exists(lmdb_path):
            os.makedirs(lmdb_path)
        if write:
            self._lmdb_env = lmdb.open(
                lmdb_path, map_size=1099511627776
            ) # 打开环境 可读可写
            self._lmdb_txn = self._lmdb_env.begin(write=True) # 建立事务
        else:
            self._lmdb_env = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=True
            ) # 打开环境 只读模式
            self._lmdb_txn = self._lmdb_env.begin(write=False) # 建立事务
        # print('Load lmdb length:{}.'.format(len(self.keys())))

    def __getitem__(self, key_name):
        """
        根据 key_name 获取图片数据 注意是 RGB 图片
        """
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        try:
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            data = torchvision.io.decode_image(image_buf, mode=torchvision.io.ImageReadMode.RGB)
        except:
            data = torch.load(io.BytesIO(payload), weights_only=True)
        return data

    def __del__(self,):
        if not self._manual_close:
            warn('Writing engine not mannuly closed!', RuntimeWarning)
            self.close()

    def load(self, key_name, type='image', **kwargs):
        """
        还是根据 key_name 获取图片数据
        """
        assert type in ['image', 'torch']
        payload = self._lmdb_txn.get(key_name.encode())
        if payload is None:
            raise KeyError('Key:{} Not Found!'.format(key_name))
        if type == 'torch':
            torch_data = torch.load(io.BytesIO(payload), weights_only=True)
            return torch_data
        elif type == 'image':
            image_buf = torch.tensor(np.frombuffer(payload, dtype=np.uint8))
            # 适应各种图片格式
            if 'mode' in kwargs.keys():
                if kwargs['mode'].lower() == 'rgb':
                    _mode = torchvision.io.ImageReadMode.RGB
                elif kwargs['mode'].lower() == 'rgba':
                    _mode = torchvision.io.ImageReadMode.RGB_ALPHA
                elif kwargs['mode'].lower() == 'gray':
                    _mode = torchvision.io.ImageReadMode.GRAY
                elif kwargs['mode'].lower() == 'graya':
                    _mode = torchvision.io.ImageReadMode.GRAY_ALPHA
                else:
                    raise NotImplementedError
            else:
                _mode = torchvision.io.ImageReadMode.RGB
            image_data = torchvision.io.decode_image(image_buf, mode=_mode)
            return image_data
        else:
            raise NotImplementedError

    def dump(self, key_name, payload, type='image', encode_jpeg=True):
        """
        key_name: str, key name to store the data.
        payload: torch.Tensor or dict, data to be stored.
        type: str, type of the data. 'image' or 'torch'.
        encode_jpeg: bool, whether to encode the image as jpeg.

        注意:
        self._lmdb_txn.put(key_name.encode(), payload_encoded)
        保存的key_name是字节类型的,payload_encoded是字节类型的
        """
        assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        assert type in ['image', 'torch']
        # 如果 key_name 已经存在，直接返回
        if self._lmdb_txn.get(key_name.encode()):
            print('Key:{} exsists!'.format(key_name))
            return 
        if type == 'torch': # 如果是 torch.Tensor 或者 dict
            assert isinstance(payload, torch.Tensor) or isinstance(payload, dict), payload
            torch_buf = io.BytesIO()
            if isinstance(payload, torch.Tensor):
                torch.save(payload.detach().float().cpu(), torch_buf)
            else:
                for key in payload.keys():
                    payload[key] = payload[key].detach().float().cpu()
                torch.save(payload, torch_buf)
            payload_encoded = torch_buf.getvalue()
            # torch_data = torch.load(io.BytesIO(payload_encoded), weights_only=True)
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        elif type == 'image':
            assert payload.dim() == 3 and payload.shape[0] == 3 # 三通道 RGB 图片
            if payload.max() < 2.0:
                print('Image Payload Should be [0, 255].')
            if encode_jpeg: # jpg 或者 jpeg 图片
                payload_encoded = torchvision.io.encode_jpeg(payload.to(torch.uint8), quality=95)
            else: # png 图片
                payload_encoded = torchvision.io.encode_png(payload.to(torch.uint8))
            """
            lambda x: int.to_bytes(x, 1, 'little')。这个lambda函数将每个整数转换为单字节的二进制表示，并使用小端字节序（little-endian）。参数1指定了使用一个字节来表示每个整数，适合存储0-255范围内的值（如图像像素值）

            payload_encoded.numpy().tolist() 将张量转换为numpy数组，然后转换为列表。这样做是为了将数据从PyTorch张量转换为Python列表，以便可以使用map函数进行处理。

            b'': 字节序列
            """
            payload_encoded = b''.join(map(lambda x:int.to_bytes(x,1,'little'), payload_encoded.numpy().tolist()))
            self._lmdb_txn.put(key_name.encode(), payload_encoded)
        else:
            raise NotImplementedError
        self._dump_counter += 1
        # 每 2000 次提交一次事务，避免内存占用过大
        if self._dump_counter % 2000 == 0: 
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def exists(self, key_name):
        """
        查询是否有 key_name 对应的图片数据
        """
        if self._lmdb_txn.get(key_name.encode()):
            return True
        else:
            return False

    def delete(self, key_name):
        """
        删除 key_name 对应的数据
        """
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not self.exists(key_name):
            print('Key:{} Not Found!'.format(key_name))
            return
        deleted = self._lmdb_txn.delete(key_name.encode())
        if not deleted:
            print('Delete Failed: {}!'.format(key_name))
            return
        self._lmdb_txn.commit()
        self._lmdb_txn = self._lmdb_env.begin(write=True)

    def raw_load(self, key_name, ):
        raw_payload = self._lmdb_txn.get(key_name.encode())
        return raw_payload

    def raw_dump(self, key_name, raw_payload):
        if not self._write:
            raise AssertionError('Engine Not Running in Write Mode.')
        if not hasattr(self, '_dump_counter'):
            self._dump_counter = 0
        self._lmdb_txn.put(key_name.encode(), raw_payload)
        self._dump_counter += 1
        if self._dump_counter % 2000 == 0:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)

    def keys(self, ):
        """
        获取所有的 key_name
        注意: 这里的 key_name 是字节类型的, 需要 decode 成字符串类型
        """
        all_keys = list(self._lmdb_txn.cursor().iternext(values=False))
        all_keys = [key.decode() for key in all_keys]
        # print('Found data, length:{}.'.format(len(all_keys)))
        return all_keys

    def close(self, ):
        """
        关闭 lmdb 数据库
        注意: 这里的 lmdb 数据库是可读可写的, 需要手动提交事务
        """
        if self._write:
            self._lmdb_txn.commit()
            self._lmdb_txn = self._lmdb_env.begin(write=True)
        self._lmdb_env.close()
        self._manual_close = True

    def random_visualize(self, vis_path, k=15, filter_key=None):
        """
        随机可视化一些图片
        """
        all_keys = self.keys()
        if filter_key is not None:
            all_keys = [key for key in all_keys if filter_key in key]
        all_keys = random.choices(all_keys, k=k)
        print('Visualize: ', all_keys)
        images = [self.load(key, type='image')/255.0 for key in all_keys]
        images = [torchvision.transforms.functional.resize(i, (256, 256), antialias=True) for i in images]
        torchvision.utils.save_image(images, vis_path, nrow=5)
