import os
import requests
from dotenv import load_dotenv

load_dotenv()

class Genie100Handler:
    def __init__(self, workspace, token, space_id):
        self.workspace = workspace
        self.token = token
        self.space_id = space_id
        self.headers = {'Authorization': f'Bearer {self.token}'}

    def _run_api(self, url, method='GET', data_json=None):
        response = requests.request(method=method, url=url, headers=self.headers, json=data_json)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        return response.json()

    def start_conversation(self, question):
        url = f"https://{self.workspace}/api/2.0/genie/spaces/{self.space_id}/start-conversation"
        return self._run_api(url, method='POST', data_json={"content": question})

    def ask_followup(self, conversation_id, question):
        url = f"https://{self.workspace}/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages"
        return self._run_api(url, method='POST', data_json={"content": question})

    def get_query_info(self, conversation_id, message_id):
        url = f"https://{self.workspace}/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}"
        return self._run_api(url)

    def get_query_result(self, conversation_id, message_id, attachment_id):
        url = f"https://{self.workspace}/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}/query-result/{attachment_id}"
        return self._run_api(url)
