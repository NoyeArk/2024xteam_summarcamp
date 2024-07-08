import requests
import json


def main():
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImgv2?access_token=24.41db67a160c389e2c97f6a4a9c5bb904.2592000.1722842786.282335-91375659"

    payload = json.dumps({
        "task_id": "1809502773683044222"
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    main()
