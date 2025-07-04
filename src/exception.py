# import sys
# from logger import logging

# # logging.basicConfig(
# #     filename='error.log',
# #     filemode='a',  # Append mode (use 'w' to overwrite)
# #     level=logging.INFO,
# #     format='%(asctime)s - %(levelname)s - %(message)s'
# # )
# def error_message_detail(error,error_detail:sys):
#     _,_,exc_tb=error_detail.exc_info()
#     file_name=exc_tb.tb_frame.f_code.co_filename
#     error_message="Error has occured in python script name [{0}] line number [{1}] error message [{2}]".format(
#         file_name,exc_tb.tb_lineno,str(error))

#     return error_message


# class CustomException(Exception):
#     def __init__(self,error_message,error_detail:sys):
#         super().__init__(error_message)
#         self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
#     def __str__(self):
#         return self.error_message
    
# if __name__=='__main__':
#     try:
#         a=1/0
#     except ZeroDivisionError as e:
#         logging.info("Divide by Zero")
#         err= CustomException(e,sys)
#         print(err)
#         raise err
import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message