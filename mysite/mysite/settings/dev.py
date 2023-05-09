from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-0jyf9v2xjgp@@&)f@-k__^^lr+!$g8pk1a5q(+u*^g8ml%_#$c"

# SECURITY WARNING: define the correct hosts in production!
ALLOWED_HOSTS = ["*"]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

INSTALLED_APPS = INSTALLED_APPS +  [
    "wagtail.contrib.styleguide",  
    ]

try:
    from .local import *
except ImportError:
    pass
