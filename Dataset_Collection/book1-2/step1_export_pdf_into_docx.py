import os
import binascii
import httpx
import json
from docx import Document

#这个API对上传PDF大小有限制，应该是100MB，建议直接在Adobe中导出
def export_pdf_into_docx(pdf_file_path,client_id,client_secret,save_file_path):
    #Generate token:
    print("Trying to generate access token...")
    content_type={'Content-Type':'application/x-www-form-urlencoded'}
    token_response=httpx.post(url='https://pdf-services.adobe.io/token',headers=content_type,data={"client_id":client_id,"client_secret":client_secret})
    access_token=json.loads(token_response.content)['access_token']
    print(f"{token_response.__str__()}\naccess_token:{access_token}+\n")

    #Get upload pre-signed URL:
    print("Trying to get upload pre-signed URL...")
    upload_header={'X-API-Key':client_id,'Authorization':'Bearer '+access_token,'Content-Type':'application/json'}
    upload_response=httpx.post(url='https://pdf-services.adobe.io/assets',headers=upload_header,data='{"mediaType": "application/pdf"}')
    upload_url=json.loads(upload_response.content)['uploadUri']
    asset_id=json.loads(upload_response.content)['assetID']
    print(f"{upload_response.__str__()}\nupload_url:{upload_url}\nasset_id:{asset_id}\n")

    #Upload the pdf file:
    print("Trying to upload the pdf file...")
    upload_pdf_file=open(pdf_file_path,'rb')
    pdf_body={'assetID':asset_id,'getCharBounds':False,}
    upload_asset_response=httpx.put(url=upload_url,headers={'Content-Type': 'application/pdf' },files={'file':upload_pdf_file},follow_redirects=True)
    print(f'{upload_asset_response.__str__()}\n{upload_asset_response.headers}')
    print('\n')

    #Export a PDF file into docx:
    export_pdf_headers={'X-API-Key':client_id,'Authorization':'Bearer '+access_token}
    export_pdf_request_body={"assetID": asset_id,"targetFormat": "docx","ocrLang": "zh-CN"}
    export_response=httpx.post(url='https://pdf-services-ue1.adobe.io/operation/exportpdf',headers=export_pdf_headers,json=export_pdf_request_body)

    #Poll the ocr job for completion:
    print('Polling the ocr job for completion...' )
    export_response_jobID=export_response.headers['x-request-id']
    export_response_location=export_response.headers['location']
    while(True):
        polled_result=httpx.get(url=f'https://pdf-services-ue1.adobe.io/operation/exportpdf/{export_response_jobID}/status',headers=export_pdf_headers)
        
        export_pdf_job_status=json.loads(polled_result.read())["status"]
        print(f'polled result:{polled_result.__str__()}\n polled response header:{polled_result.headers.__str__()}\n\
              pdf ocr job status: {export_pdf_job_status}')
        
        if(polled_result.status_code==200):
            if(export_pdf_job_status=='done'):
                download_Uri=json.loads(polled_result.content)['asset']['downloadUri']
                #print(download_Uri)
                print(polled_result.content)
                download_file(download_Uri, save_file_path)
                break
            if(export_pdf_job_status=='failed'):
                print("The job is failed.")
                print(polled_result.content)
                break

def download_file(download_uri, target_path):
    with httpx.stream("GET", download_uri) as response:
        if response.status_code == 200:
            with open(target_path, 'wb') as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        else:
            print(f"Unable to download the file. Status code: {response.status_code}")

if(__name__=='__main__'):
    client_id='your_id'
    client_secret='your_secret'
    upload_pdf_path='../book2_pair_yya/CFP/'    # replace   
    upload_pdf_file='output_CFP.pdf'            # replace
    save_file_path='../book2_pair_yya/CFP/'     # replace 
    save_docx_file='output_CFP.docx'            # replace
    export_pdf_into_docx(upload_pdf_path+upload_pdf_file,client_id,client_secret,save_file_path+save_docx_file)

    