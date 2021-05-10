from gazeclassify.core.services.analysis import FrameResult, Classification, JsonSerializer

#
# def test_ExportJson():
#     results = [
#         FrameResult([Classification("person", 1), Classification("person", 2)]),
#         FrameResult([Classification("background", 1), Classification("background", 1)])
#     ]
#
#     serializer = JsonSerializer()
#     serializer.encode(results, "try.json")