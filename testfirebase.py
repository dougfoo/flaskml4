from google.cloud import firestore
import os
import datetime


def firestore_fetch(max=50):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gkey.json"
    db = firestore.Client()
    # doc_ref = db.collection(u'queries')
    # doc_ref.set({
    # 	u'text': u'Ada i love you',
    # 	u'from': u'Anonymouse',
    # 	u'date': u'2020-01-27T00:43:58.021920'
    # })

    # Then query for documents
    users_ref = db.collection(u'queries').order_by(u'date', direction=firestore.Query.DESCENDING).limit(max)

    buf = []
    for doc in users_ref.stream():
        buf.append(doc.to_dict())
        print(u'{} => {}'.format(doc.id, doc.to_dict()))
    return buf


def firestore_add(txt, fr='anon', dt=datetime.datetime.now()):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gkey.json"
    db = firestore.Client()
    doc_ref = db.collection(u'queries')
    doc_ref.add({
        u'text': txt,
        u'from': fr,
        u'date': dt.isoformat("T")
    })

print(firestore_add('msg23'))
print(firestore_fetch())
