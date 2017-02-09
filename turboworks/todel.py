from turboworks.reference import ref_dict

z = {'hi': 3, 'ok':{'doe':{'hey':{'kid':{'smoke':{'this':1}}}}}}


delimiter = '.'

def key_scavenger(d, compound_key = '', top_level = True, compound_keys = None):
    # returns all highest level non-dict entries in a list of class/attr dict style strings
    # will be used as a tool to return all values if a compound key ends in a dict.
    # for example selecting 'types.old' from ref_dict results in selecting 'types.old.q'
    # and 'types.old.r'

    if compound_keys is None:
        compound_keys = []

    for key in d:
        if type(d[key]) is dict:
            if top_level:
                key_scavenger(d[key], compound_key=compound_key + key,
                              top_level=False, compound_keys=compound_keys)
            else:
                key_scavenger(d[key], compound_key=compound_key + delimiter + key,
                                  top_level=False, compound_keys=compound_keys)

        else:
            if top_level:
                compound_keys.append(key)
            else:
                compound_keys.append(compound_key + delimiter + key)

    if top_level:
        return compound_keys

print key_scavenger(z)