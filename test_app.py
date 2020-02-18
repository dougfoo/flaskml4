import app


def test_tokenize():
    assert ' what a world '.strip() == 'what a world'


def test_azure():
    resp = app.azure_sentiment('i love you')
    print(resp)
    assert(round(float(resp['nScore']), 2) == 0.96)
