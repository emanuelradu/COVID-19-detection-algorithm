def purgeChar(c):
    if   c == 'ă':
        return 'a'
    elif c == 'Ă':
        return 'A'
    elif c == 'î':
        return 'i'
    elif c == 'Î':
        return 'I'
    elif c == 'ș':
        return 's'
    elif c == 'Ș':
        return 'S'
    elif c == 'ț':
        return 't'
    elif c == 'Ț':
        return 'T'
    elif c == 'â':
        return 'a'
    elif c == 'Â':
        return 'A'
    return c

def purge(data):
    return ''.join(map(purgeChar, data)).lower()
