import sys
import logging

def error_message_details(error, error_detail: sys):
    _, _, exc_tbl = error_detail.exc_info()
    filename = exc_tbl.tb_frame.f_code.co_filename
    return f"Error occured in python script {filename} line number {exc_tbl.tb_lineno} error message {str(error)}"


class CustomException(Exception): 
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message= error_message_details(error_message, error_detail=error_detail)

    
    def __str__(self) -> str:
        return self.error_message
