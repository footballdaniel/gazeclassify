# from gazeclassify.core.service.analysis import FrameResult, Classification, JsonSerializer
#
#
# def test_ExportJson():
#     results = [
#         FrameResult([Classification("person", 1), Classification("person", 2)]),
#         FrameResult([Classification("background", 1), Classification("background", 1)])
#     ]
#
#     eyetracker = JsonSerializer()
#     eyetracker.encode(results, "try.json")