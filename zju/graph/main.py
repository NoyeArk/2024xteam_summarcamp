import requests
import json

API_KEY = "1zvVmImc5oVwyAZDvWJkD2Fz"
SECRET_KEY = "de5UGEFZ4CnhO4IiG1RRC4eEc7bS4ccP"


def main():
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/text2image/sd_xl?access_token=" + get_access_token()

    payload = json.dumps({
        "prompt": "机器人可以具有不同的形态和功能，从简单的自动化设备到复杂的类人机器人。它们可以用于制造业、服务业、医疗、军事、科学研究等多个领域。例如，工业机器人可以用于装配线上的重复性任务，而服务机器人则可以在酒店、餐厅、家庭中提供各种服务。",
        "size": "1024x1024",
        "n": 1,
        "steps": 20,
        "sampler_index": "Euler a"
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
