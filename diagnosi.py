from experta import *


def valid_response(response: str):
    valid = False
    response = response.lower()
    if response == "si" or response == "no":
        valid = True
    return valid

