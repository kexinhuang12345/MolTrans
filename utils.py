import datetime
import string
import random

def get_date_postfix():
    """Get a date based postfix for directory name"""
    dt = datetime.datetime.now()
    post_fix = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(dt.year,dt.month, dt.day,  dt.hour, dt.minute, dt.second)
    return post_fix


def random_string(string_len=3):
    """Get a random string"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))
