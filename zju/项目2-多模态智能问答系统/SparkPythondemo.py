# coding: utf-8
import SparkApi

from SparkApi import main

# 以下密钥信息从控制台获取   https://console.xfyun.cn/services/bm35
appid = "698c8614"  # 填写控制台中获取的 APPID 信息
api_secret = "NDJhOTIzODljY2ZiMDk4OTg5MmZjYzI4"  # 填写控制台中获取的 APISecret 信息
api_key = "2fb3adb3314e207f14c4fce53e82b67a"  # 填写控制台中获取的 APIKey 信息

domain = '4.0Ultra'  # Max版本
# domain = "generalv3"       # Pro版本
# domain = "general"         # Lite版本

Spark_url = 'wss://spark-api.xf-yun.com/v4.0/chat'  # Max服务地址
# Spark_url = "wss://spark-api.xf-yun.com/v3.1/chat"  # Pro服务地址
# Spark_url = "wss://spark-api.xf-yun.com/v1.1/chat"  # Lite服务地址

# 初始上下文内容，当前可传system、user、assistant 等角色
text = [
    # {"role": "system", "content": "你现在扮演李白，你豪情万丈，狂放不羁；接下来请用李白的口吻和用户对话。"} , # 设置对话背景或者模型角色
    # {"role": "user", "content": "你是谁"},  # 用户的历史问题
    # {"role": "assistant", "content": "....."} , # AI的历史回答结果
    # # ....... 省略的历史对话
    # {"role": "user", "content": "你会做什么"}  # 最新的一条问题，如无需上下文，可只传最新一条问题
]


def getText(role, content):
    jsoncon = {"role": role, "content": content}
    text.append(jsoncon)
    return text


def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length


def checklen(text):
    while getlength(text) > 8000:
        del text[0]
    return text


if __name__ == '__main__':
    while 1:
        Input = input("\n" + "我:")
        if Input == "exit":
            print("诺亚方舟:下次再见，祝您生活愉快！")
            break
        question = checklen(getText("user", Input))
        answer = ""
        print("诺亚方舟:", end="")
        main(appid, api_key, api_secret, Spark_url, domain, question)
        # print(SparkApi.answer)
        getText("assistant", answer)

