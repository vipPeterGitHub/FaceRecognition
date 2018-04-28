# -*- coding:utf-8 -*-
import urllib
import urllib2
import re
import os
import thread
import time



def fetch_pictures(url):
    html_content = urllib.urlopen(url).read()
    r = re.compile('<img pic_type="0" class="BDE_Image" src="(.*?)"')
    picture_url_list = r.findall(html_content.decode('utf-8'))

    os.mkdir('pictures')
    os.chdir(os.path.join(os.getcwd(), 'pictures'))
    for i in range(len(picture_url_list)):
        picture_name = str(i) + '.jpg'
        try:
            urllib.request.urlretrieve(picture_url_list[i], picture_name)
            print("Success to download " + picture_url_list[i])
        except:
            print("Fail to download " + picture_url_list[i])

def getHtml(url):
	page = urllib.urlopen(url)
	html = page.read()
	return html

def getImg(html):
	reg=r'"objURL":"(.*?)"'
	#reg = r'<img class="mimg" style="background-color:#\w\w\w\w\w\w;color:#\w\w\w\w\w\w" height="\d\d\d" width="\d\d\d" src="(.*?)"'
	imgre=re.compile(reg)
	imglist = re.findall(imgre,html)
	return imglist

def download(urls,path,index):
	for url in urls:
		try:
			res = urllib2.Request(url)
			if str(res.status_code)[0]=="4":
				continue
		except Exception as e:
			print ("failed")
		filename = os.path.join(path,str(index)+".jpg")
		print filename
		try:
			urllib.urlretrieve(url,filename)
			index+=1
		except:
			print ("save failed!")
			index+=1

#Savepath = "D:/pachong/"
foldername = 'renlian'
totalnum = 60

os.mkdir(foldername)
Savepath = "D:/face_recognition/testpy/"+foldername+"/"
for index in range(0,totalnum,30):
	#index = 0
	#rootHtml = "https://image.baidu.com/search/index?tn=baiduimage&ipn=2&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1524325418094_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E8%84%B8&pn="+str(index)
	#rootHtml = "https://image.baidu.com/search/index?ct=201326592&cl=2&st=-1&lm=-1&nc=1&ie=utf-8&tn=baiduimage&ipn=r&rps=1&pv=&fm=rs1&word=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E8%84%B8%E5%9B%BE%E7%89%87%E5%BA%93%E4%BE%A7%E9%9D%A2&oriquery=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E8%84%B8%E5%9B%BE%E7%89%87%E5%BA%93&ofr=%E4%B8%AD%E5%9B%BD%E4%BA%BA%E8%84%B8%E5%9B%BE%E7%89%87%E5%BA%93&sensitive=0"+str(index)
	#rootHtml = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1524328318666_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E6%AD%8C%E6%98%9F%E5%91%A8%E6%9D%B0%E4%BC%A6"+str(index)
	rootHtml = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1524328527087_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&word=%E4%BA%BA%E8%84%B8"+str(index)
	download(getImg(getHtml(rootHtml)),Savepath,index)