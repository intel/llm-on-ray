import ipaddress
import random
import secrets
import string



# option 1: similar to bigscience-pii redaction:
#         # replace with [TAG], e.g., [EMAIL]
#         #redacted_str, metadata = redact_pii(text, matches)
# option 2: similar to bigcode-pii redaction:
#         # IP: replace with predefined random IP address, or DNS servers
#         # EMAIL, USERNAME, KEY: replace with random values
#         # also keeping track of pii values through a sample
#         # and replace with the same random value for the same pii value
#         #print(redacted_str)
#         redacted_str = redact_pii_with_random_values(text, matches)
#         # metadata_out = {"regex metadata":metadata, "original": text, "redacted": redacted_str}
#         # match_set = (redacted_str, metadata_out)


# The IP replacements are copied from bigcode-pii
# List of random private IP addresses to use as replacements
REPLACEMENTS_IP = {
    "IPv4": ["172.16.31.10", "172.16.58.3", "172.16.17.32", "192.168.127.12", "192.168.3.11"],
    "IPv6": [
        "fd00:c2b6:b24b:be67:2827:688d:e6a1:6a3b",
        "fd00:a516:7c1b:17cd:6d81:2137:bd2a:2c5b",
        "fc00:e968:6179::de52:7100",
        "fc00:db20:35b:7399::5",
        "fdf8:f53e:61e4::18",
    ],
}

# providergs = ["google", "cloudfare", "alternate-dns", "quad9","open-dns", "comodo", "adguard"]
POPULAR_DNS_SERVERS = [
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1",
    "1.0.0.1",
    "76.76.19.19",
    "76.223.122.150",
    "9.9.9.9",
    "149.112.112.112",
    "208.67.222.222",
    "208.67.220.220",
    "8.26.56.26",
    "8.20.247.20",
    "94.140.14.14",
    "94.140.15.15",
]

letters = string.ascii_lowercase
digits = string.digits
lettters_digits = string.ascii_lowercase + string.digits

# random emails
n = 100
REPLACEMENT_EMAIL = [
        "".join(secrets.choice(letters) for i in range(10)) + "@example.com"
        for i in range(n)
    ]

# random keys
REPLACEMENT_KEY = [
        "".join(secrets.choice(digits) for i in range(10))
        for i in range(n)
    ]
# simple hack: make key replacement and phone replacement to be 
# both 10 random digits
# to simplify redaction
# [
#         "".join(secrets.choice(lettters_digits) for i in range(32)) for i in range(n)
#     ]

# random usernames
REPLACEMENT_USERNAME = [
        "@"+"".join(secrets.choice(letters) for i in range(10))
        for i in range(n)
    ]

REPLACEMENT_PHONE = [
        "".join(secrets.choice(digits) for i in range(10))
        for i in range(n)
    ]

REPLACEMENT_DICT={
    'EMAIL': REPLACEMENT_EMAIL,
    'KEY': REPLACEMENT_KEY,
    'USER': REPLACEMENT_USERNAME,
    'PHONE_NUMBER':REPLACEMENT_PHONE
}

def is_private_ip(ip):
    """Check if an IP address is allocated for private networks"""
    ip = ipaddress.ip_address(ip)
    return ip.is_private

def replace_ip(value):
    """Replace an IP address with a synthetic IP address of the same format"""
    # ipaddress.ip_address(ip) raises exception when ip i snot valid
    # if is_private_ip(value) or (value in POPULAR_DNS_SERVERS):
    #     return value
    
    if value in POPULAR_DNS_SERVERS:
        print('IP is one of DNS servers, return original value: ', value)
        return value

    try:
        ipaddress.IPv4Address(value)
        print('IP is IPv4, return redacted value')
        return secrets.choice(REPLACEMENTS_IP["IPv4"])
    except ValueError:
        try:
            ipaddress.IPv6Address(value)
            print('IP is IPv6, return redacted value')
            return secrets.choice(REPLACEMENTS_IP["IPv6"])
        except ValueError:
            # this doesn't happen if we already use ipaddress filter in the detection
            # this is good as we have another layer of protection to redace false positive
            print("Invalid IP address:", value)
            return value

def redact_email_key_user_phone(value, tag):
    supported_tags = {'KEY', 'EMAIL', 'USER', 'PHONE_NUMBER'}
    if tag in supported_tags:
        #return secrets.choice(REPLACEMENT_DICT[tag]) 
        if tag=='KEY':
            redact_value = "".join(secrets.choice(digits) for i in range(10))
        if tag == 'EMAIL':
            redact_value = "".join(secrets.choice(letters) for i in range(10)) + "@{}.com".format("".join(secrets.choice(letters) for i in range(5)))
        if tag == 'USER':
            redact_value = "@"+"".join(secrets.choice(letters) for i in range(10))
        if tag == 'PHONE_NUMBER':
            redact_value = "".join(secrets.choice(digits) for i in range(10))
        return redact_value
    else:
        print('{} type is not supported!'.format(tag))
        return value 


# TODO: generate random strings on the fly, instead of choose from one of n
def redact_pii_with_random_values(text, matches):
  # adapted from bigcode-pii redaction
  # however, matches here is a list of dictionaries
  # the dictionary is of this schema:
  # {'start': 123, 'end': 234, 'value': xyz, 'type': PHONE_NUMBER}
    redacted_str = text
    replaced_values = []
    lookup_dict = {}
    for match in matches:
        start_idx = match['start']
        end_idx = match['end']
        matched_str = match['value'] #text[start_idx:end_idx]
        tag = match['type']
        if matched_str in replaced_values:
            redact_tag = lookup_dict[matched_str]
        else:
            if tag == 'IP_ADDRESS':   
                redact_tag = replace_ip(matched_str)
                
            else:
                redact_tag = redact_email_key_user_phone(matched_str, tag)

            replaced_values.append(matched_str)
            lookup_dict[matched_str]=redact_tag
        
        # print('original: ', matched_str)
        # print('redacted tag: ', redact_tag)
        match['redacted'] = redact_tag
        redacted_str = redacted_str.replace(matched_str, redact_tag)
    # Create the "metadata" as all of the information we had before redaction
    #metadata += [(match)]
    print(matches)
    return redacted_str


def redact_pii_with_tags(text, matches):
    # adapted from bigscience-pii
    redacted_str = text
    for match in matches:
        matched_str = match['value']
        tag = match['type']
        redact_tag = "[" + tag +"]"
        redacted_str = redacted_str.replace(matched_str, redact_tag)

    return redacted_str
