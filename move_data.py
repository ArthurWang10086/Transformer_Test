# -*- coding: utf-8 -*-
# /usr/bin/python2

import os

normal_start_dir = "/home/zhoujialiang/workspace/dataset/20180830-20180920_sanhaun_only/normal/"
normal_end_dir_train = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/normal_train/"
normal_end_dir_test = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/normal_test/"

plugin_start_dir = "/home/zhoujialiang/workspace/dataset/20180830-20180920_sanhaun_only/waigua/"
plugin_end_dir_train = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/plugin_train/"
plugin_end_dir_test = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/plugin_test/"

merge_train = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/train/"
merge_test = "/home/luoyifan/dataset/nsh/sanhuangua/transformer/v_2/dataset/test/"

def dir_check(path):
    if not os.path.exists(path):
        os.mkdir(path)

def copy_file(start_dir,end_dir_train,end_dir_test):
	count = 0
	dir_check(end_dir_train)
	dir_check(end_dir_test)
	for filename in os.listdir(start_dir):
		count+=1
		if(count < 8000):
			os.system('cp '+start_dir+str(filename)+' '+end_dir_train)
		else:
			os.system('cp '+start_dir+str(filename)+' '+end_dir_test)
		if(count%1000 == 0):
			print(count)
		if(count == 10000):
			break
if __name__ == '__main__':
	copy_file(normal_start_dir,normal_end_dir_train,normal_end_dir_test)
	copy_file(plugin_start_dir,plugin_end_dir_train,plugin_end_dir_test)

	dir_check(merge_train)
	dir_check(merge_test)

	os.system('cp '+normal_end_dir_train+'* '+merge_train)
	os.system('cp '+plugin_end_dir_train+'* '+merge_train)

	os.system('cp '+normal_end_dir_test+'* '+merge_test)
	os.system('cp '+plugin_end_dir_test+'* '+merge_test)