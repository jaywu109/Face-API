import json
import os
import time
from datetime import datetime

from flask import abort

from . import license_util


class VerifyError(Exception):
    pass


class UsageError(Exception):
    pass


service_expiry = None
license_file_path = "/wisers/license.json"  # fixed path


def is_service_expired():
    if not service_expiry:
        return False
    service_expiry_ts = service_expiry.timestamp()
    now = datetime.now()
    now_ts = now.timestamp()
    res = now_ts > service_expiry_ts
    if res:
        print(f"!!! Service expired (service expiry: {service_expiry})")
    return res


def verify_call(func):
    def inner(*args, **kwargs):
        if is_service_expired():
            abort(403)
        else:
            return func(*args, **kwargs)

    return inner


def verify():
    # get and parse ARGUMENT_JSON
    try:
        argument_json = os.environ["ARGUMENT_JSON"]
        argument_json = license_util.decrypt(argument_json)
    except:
        raise VerifyError("unable to understand ARGUMENT_JSON")

    print(f"ARGUMENT_JSON={argument_json}")
    run_data = json.loads(argument_json)
    #     print(f"run_data={run_data}")

    # get data passed in via ARGUMENT_JSON
    license_secret_sign = run_data["sign"]
    host_mac = run_data["mac"]
    host_tz = run_data["tz"]
    host_utcnow = run_data["utcnow"]
    host_now_ts = run_data["now-ts"]

    # check datetime
    utcnow = datetime.utcnow()
    now_ts = datetime.now().timestamp()
    diff = abs(host_now_ts - now_ts)
    print(f"UTC [{utcnow}] vs host UTC [{host_utcnow}]")
    print(f"TZ [{time.tzname[time.daylight]}] vs host TZ [{host_tz}]")
    print(f"now-ts [{now_ts}] vs now-ts [{host_now_ts}]")

    if diff < 300:
        print(f"current system time is only {diff}s away from startup by host")
    else:
        raise VerifyError(f"it has been a while ({diff}s) since startup by host")

    # read file
    license_info = license_util.read_license(license_file_path)

    if not license_info:
        raise VerifyError("failed to read file")
    print(f"license_info={license_info}")

    # verify file signature
    if license_util.verify_license(license_info, license_secret_sign):
        print("security file signature verified")
    else:
        raise VerifyError("failed to verify file signature")

    # check mac
    allowed_macs = license_info["macs"]
    if host_mac in allowed_macs:
        print(f"*** host mac {host_mac} is one of allowed {allowed_macs}")
    else:
        raise VerifyError(f"host {host_mac} not allowed")

    global service_expiry
    service_expiry = datetime.fromisoformat(license_info["service-expiry"])

    if is_service_expired():
        raise UsageError(f"service expired (service expiry: {service_expiry})")
