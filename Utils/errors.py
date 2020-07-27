class NonAligned(Error):
    print('The directories provided do not contain aligned data')
    pass

class DoesNotContain(Error):
    print('A directory provided does not contain folders "real" and/or "fake"')
    pass