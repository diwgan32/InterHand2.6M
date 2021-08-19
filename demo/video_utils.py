import skvideo

skvideo.setFFmpegPath("/usr/bin")
import skvideo.io


def getFrameRate(video_filename):
    try:
        metadata = skvideo.io.ffprobe(video_filename)
        frameRateStr = metadata["video"]["@avg_frame_rate"]
        frameRate = 1.0 * int(frameRateStr.split("/")[0]) / int(frameRateStr.split("/")[1])
        return frameRate
    except:
        return 30.0

def getUserReportedMetadata(video_filename):
    retobj = {}
    metadata = skvideo.io.ffprobe(video_filename)
    retobj["fps"] = round(getFrameRate(video_filename))
    retobj["duration"] = float(metadata["video"]["@duration"])
    retobj["isPortrait"] = isVideoPortrait(video_filename)
    return retobj


def getAllMetadata(video_filename):
    retobj = getUserReportedMetadata(video_filename)
    retobj["dims"] = getDims(video_filename)
    return retobj


def getVideoRotate(video_filename):
    try:
        metadata = skvideo.io.ffprobe(video_filename)
        tags = metadata["video"]["tag"]
        for i in range(len(tags)):
            keyName = tags[i]["@key"]
            if (keyName == "rotate"):
                try:
                    keyVal = int(tags[i]["@value"])
                    return keyVal
                except:
                    continue
        return 0
    except Exception as e:
        print(e)
        return 0

def isUpsideDown(video_filename):
    rotate = getVideoRotate(video_filename)
    return rotate == 180 or rotate == -180

def isVideoPortrait(video_filename):
    rotate = getVideoRotate(video_filename)
    return rotate == 90 or rotate == 270


def getDims(video_filename):
    metadata = skvideo.io.ffprobe(video_filename)
    width = metadata["video"]["@width"]
    height = metadata["video"]["@height"]
    if (isVideoPortrait(video_filename)):
        return (int(width), int(height))
    else:
        return (int(height), int(width))
