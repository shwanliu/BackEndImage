import collections
import json

def JsonResult(data,code,status):
        result = collections.OrderedDict()
        result["code"]=code
        result["status"]=status
        result["data"]=data
        return json.dumps(result)