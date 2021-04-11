import cv2
from moviepy.editor import *
import numpy as np
import sys
import json
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


CONFIG_JSON_FILENAME = 'config.json'
# HSV:
# H:   0 - 180
# S,V: 0 - 255
COLOR_RED_LOWER = np.array([170, 150, 175])
COLOR_RED_UPPER = np.array([180, 255, 255])
COLOR_RED2_LOWER = np.array([0, 150, 175])
COLOR_RED2_UPPER = np.array([10, 255, 255])
COLOR_WHITE_LOWER = np.array([0, 0, 200])
COLOR_WHITE_UPPER = np.array([180, 37, 255])
COLOR_RUPEE_LOWER = np.array([55, 153, 153])
COLOR_RUPEE_UPPER = np.array([60, 204, 204])

""" Gets current position of video vid in seconds """
def getPos(vid: cv2.VideoCapture):
    return vid.get(cv2.CAP_PROP_POS_MSEC) / 1000


""" Gets current position of video vid in frames """
def getFrame(vid: cv2.VideoCapture):
    return int(vid.get(cv2.CAP_PROP_POS_FRAMES))


""" Sets current position of video vid to frame frameNum """
def setFrame(vid: cv2.VideoCapture, frameNum):
    vid.set(cv2.CAP_PROP_POS_FRAMES, frameNum)


""" Gets the total number of frames in video vid """
def getTotalFrames(vid: cv2.VideoCapture):
    return int(vid.get(cv2.CAP_PROP_FRAME_COUNT))


""" Gets the total number of frames in video vid """
def getFps(vid: cv2.VideoCapture):
    return vid.get(cv2.CAP_PROP_FPS)


""" Gets the width of video vid, in pixels """
def getWidth(vid: cv2.VideoCapture):
    return vid.get(cv2.CAP_PROP_FRAME_WIDTH)


""" Gets the height of video vid, in pixels """
def getHeight(vid: cv2.VideoCapture):
    return vid.get(cv2.CAP_PROP_FRAME_HEIGHT)


""" 
Detects the location of the leftmost heart container in the video

First, it searches for a frame where the screen is entirely white (during the "Open your eyes" cutscene at the beginning 
of the game)

Then, it finds the location of the screen by searching for the biggest white rectangle with a 16:9 width-to-height ratio

The location of the heart container is (y1:y2, x1:x2) is (52:68, 88:100) for a fullscreen video.  These numbers are 
adjusted based on the proportions of the screen.

output: an image called "screen.bmp" outlining the screen in green
returns: y1, y2, x1, x2 specifying a region containing the left half of the leftmost heart container
"""
def detectHudLocation(inputVideoFilename):
    vid = cv2.VideoCapture(inputVideoFilename)
    assert vid.isOpened(), f'Could not open file "{inputVideoFilename}"'
    found = False
    while not found:
        # Skip 0.5 seconds at a time
        for i in range(int(getFps(vid) / 2) - 1):
            vid.grab()
        _, frame = vid.read()

        imgGry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgGry, 240, 255, cv2.THRESH_BINARY)

        # Find all shapes onscreen
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        biggestRect = None
        biggestArea = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(approx)
            area = abs(h * w)
            if area > biggestArea:
                biggestArea = area
                biggestRect = (x, y, w, h)
        if biggestRect is not None:
            x, y, w, h = biggestRect
            if (w * h > getHeight(vid) * getWidth(vid) / 4) and abs(h / w - 1080 / 1920) < 0.05:
                found = True
                cv2.putText(frame, "screen", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
                cv2.imwrite('screen.bmp', frame)
                heartY1 = y + int(round(h * 52 / 1080))
                heartY2 = y + int(round(h * 68 / 1080))
                heartX1 = x + int(round(w * 88 / 1920))
                heartX2 = x + int(round(w * 100 / 1920))

                # dPadY1 = y + int(round(h * 151 / 1080))
                # dPadY2 = y + int(round(h * 191 / 1080))
                # dPadX1 = x + int(round(h * 108 / 1080))
                # dPadX2 = x + int(round(h * 150 / 1080))
                #
                # rupeeY1 = y + int(round(h * 62 / 1080))
                # rupeeY2 = y + int(round(h * 88 / 1080))
                # rupeeX1 = x + int(round(h * 1778 / 1080))
                # rupeeX2 = x + int(round(h * 1792 / 1080))

                vid.release()

                return heartY1, heartY2, heartX1, heartX2


"""
return: a list of timestamps, in seconds, specifying the clips in inputVideoFilename where at least one red pixel 
is within the region heartLocation.

For example, the list [t1, t2, t3, t4] specifies two clips: one from t1 - t2, and one from t3 - t4

As this function iterates through all frames in the video, it may take a while.
"""
def detectClipends(inputVideoFilename, heartLocation):
    heartY1, heartY2, heartX1, heartX2 = heartLocation

    clipEnds = []
    vid = cv2.VideoCapture(inputVideoFilename)
    is60Fps = round(getFps(vid)) == 60  # BOTW gameplay runs at 30 FPS, so we can skip every other frame

    frameNum = 0
    endFrame = getTotalFrames(vid)

    # setFrame(vid, frameNum)
    if is60Fps:
        vid.grab()
    _, frame = vid.read()

    print(f'Total frames: {endFrame - frameNum + 1}')
    print(f'FPS: {getFps(vid)}')

    prevFrame = None
    prevMatch = 0
    
    print()
    while frame is not None and frameNum < endFrame:
        frameHeartCropped = frame[heartY1:heartY2, heartX1:heartX2]

        # frameHeartCropped = frame
        # cv2.imshow('TEST', frameHeartCropped)
        # cv2.imwrite('out.bmp', frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        hsv = cv2.cvtColor(frameHeartCropped, cv2.COLOR_BGR2HSV)
        tempMatch = cv2.inRange(hsv, COLOR_RED_LOWER, COLOR_RED_UPPER)
        tempMatch += cv2.inRange(hsv, COLOR_RED2_LOWER, COLOR_RED2_UPPER)
        heartMatch = np.nanmax(tempMatch)

        match = bool(heartMatch)

        if not prevMatch and match or prevMatch and not match:
            pos = getPos(vid)
            frame = pos * getFps(vid)
            clipEnds.append(getPos(vid))
            sys.stdout.write(f'\rFrame {getFrame(vid)}: {prevMatch} --> {match}')

        prevFrame = frame
        if is60Fps:
            vid.grab()
        _, frame = vid.read()
        prevMatch = match
        frameNum += 1

    print()
    if len(clipEnds) % 2:  # ensure length of clipEnds is even
        setFrame(vid, endFrame)
        clipEnds.append(getPos(vid))

    vid.release()

    return clipEnds


"""
Returns: list clipEnds, such that all clips specified in toInclude are contained within clipEnds

For example, with clipEnds [t1, t3, t4, t5, t7, t10] and toInclude [t2, t6, t8, t9], the returned list will be
[t1, t6, t7, t10] 
"""
def includeClipEnds(clipEnds, toInclude):
    # Must have even number of clip ends
    assert len(clipEnds) % 2 == 0
    assert len(toInclude) % 2 == 0
    if len(clipEnds) < 2:
        clipEnds.extend(toInclude)
        clipEnds.sort()
    removeClipEnds = set()
    addClipEnds = set()
    for i in range(0, len(toInclude), 2):
        includeStart = toInclude[i]
        includeEnd = toInclude[i + 1]
        if includeStart > max(clipEnds) or includeEnd < min(clipEnds):
            clipEnds.append(includeStart)
            clipEnds.append(includeEnd)
        else:
            includeIncludeStart = True
            includeIncludeEnd = True
            for j in range(0, len(clipEnds), 2):
                clipStart = clipEnds[j]
                clipEnd = clipEnds[j + 1]
                if clipStart <= includeStart and clipEnd >= includeStart:
                    includeIncludeStart = False
                if clipStart <= includeEnd and clipEnd >= includeEnd:
                    includeIncludeEnd = False
            clipEnds = [ts for ts in clipEnds if not (ts > includeStart and ts < includeEnd)]
            if includeIncludeStart:
                clipEnds.append(includeStart)
            if includeIncludeEnd:
                clipEnds.append(includeEnd)
            clipEnds.sort()

    newClipEnds = set(clipEnds)
    for ts in addClipEnds:
        newClipEnds.add(ts)
    for ts in removeClipEnds:
        newClipEnds.remove(ts)
    newClipEnds = list(newClipEnds)
    newClipEnds.sort()
    return newClipEnds


"""
Returns: list clipEnds, such that all clips specified in toExclude are removed within clipEnds

For example, with clipEnds [t1, t3, t4, t5, t7, t10] and toInclude [t2, t6, t8, t9], the returned list will be
[t1, t2, t7, t8, t9, t10] 
"""
def excludeClipEnds(clipEnds, toExclude):
    # Must have even number of clip ends
    assert len(clipEnds) % 2 == 0
    assert len(toExclude) % 2 == 0

    removeClipEnds = set()
    addClipEnds = set()
    for i in range(0, len(toExclude), 2):
        excludeStart = toExclude[i]
        excludeEnd = toExclude[i+1]
        for j in range(0, len(clipEnds), 2):
            clipStart = clipEnds[j]
            clipEnd = clipEnds[j+1]
            if clipStart >= excludeStart and clipEnd <= excludeEnd:
                removeClipEnds.add(clipStart)
                removeClipEnds.add(clipEnd)
            elif clipEnd < excludeStart or clipStart > excludeEnd:
                pass
            elif clipStart >= excludeStart and clipEnd > excludeEnd:
                removeClipEnds.add(clipStart)
                addClipEnds.add(excludeEnd)
            elif clipStart < excludeStart and clipEnd <= excludeEnd:
                removeClipEnds.add(clipEnd)
                addClipEnds.add(excludeStart)
    newClipEnds = set(clipEnds)
    for ts in addClipEnds:
        newClipEnds.add(ts)
    for ts in removeClipEnds:
        newClipEnds.remove(ts)
    newClipEnds = list(newClipEnds)
    newClipEnds.sort()
    return newClipEnds


"""
output: video file outputVideoFilename, which consists of clips clipEnds of inputVideoFilename

For example, for outputVideoFilename "test.mp4" and inputVideoFilename "input.mp4" and clipEnds [1, 2, 3, 4]
This function will output "test.mp4", which consists of the clip from 1-2 seconds and the clip from 3-4 seconds from 
"input.mp4" 
"""
def cutAndExportVideo(inputVideoFilename, outputVideoFilename, clipEnds: list):
    # Must have even number of clip ends
    assert len(clipEnds) % 2 == 0

    clips = []
    origVid = VideoFileClip(inputVideoFilename)
    print()
    for i in range(0, len(clipEnds), 2):
        start = clipEnds[i]
        end = clipEnds[i+1]
        sys.stdout.write(f'\rOpening clip {i//2 + 1} of {len(clipEnds)//2}: {start} --> {end}')
        clip = origVid.subclip(start, end)
        clips.append(clip)
    print()

    finalVid = concatenate_videoclips(clips)
    finalVid.write_videofile(outputVideoFilename, fps=30, threads=8, preset="veryfast")

    # cleanup
    for clip in clips:
        clip.close()
    finalVid.close()


if __name__ == '__main__':
    logging.info('Breath of the Wild cutscene remover')
    logging.info('by msbmteam')

    """
    PARSE config.json
    """

    logging.info(f"Reading configuration file {CONFIG_JSON_FILENAME}")

    try:
        with open(CONFIG_JSON_FILENAME, 'r') as jsonFile:
            rawString = ''.join(jsonFile)
            config = json.loads(rawString)
    except Exception as e:
        raise e.__class__('Could not open "config.json". Does it exist in the current directory?')

    VIDEO_FILENAME = config.get('inFile')
    assert VIDEO_FILENAME is not None, "inFile not found"

    VIDEO_OUT_FILENAME = config.get('outFile')
    assert VIDEO_OUT_FILENAME is not None, "outFile not found"

    assert 'timestamps' in config, "timestamps settings not found"
    assert 'detect' in config['timestamps'], "timestamps detect flag not found"
    doTsDetect = config['timestamps']['detect']

    includeTs = config['timestamps'].get('include', [])
    assert len(includeTs) % 2 == 0, 'include must have an even number of timestamps'
    excludeTs = config['timestamps'].get('exclude', [])
    assert len(excludeTs) % 2 == 0, 'exclude must have an even number of timestamps'

    logging.info(f"Configuration file parsed and loaded")

    """
    START MAIN SCRIPT
    """

    if doTsDetect:
        logging.info(f"Detecting heart container location...")
        heartLocation = detectHudLocation(VIDEO_FILENAME)
        heartY1, heartY2, heartX1, heartX2 = heartLocation
        logging.info(f'Heart container location: ({heartY1}:{heartY2}), ({heartX1}:{heartX2})')

        logging.info(f'Starting frame-by-frame analysis for heart container...')
        clipEnds = detectClipends(VIDEO_FILENAME, heartLocation)
        logging.info(f'Frame-by-frame analysis complete')
    else:
        clipEnds = []

    logging.info(f'Including specified timestamps')
    clipEnds = includeClipEnds(clipEnds, includeTs)

    logging.info(f'Excluding specified timestamps')
    clipEnds = excludeClipEnds(clipEnds, excludeTs)

    # Write clip ends to file
    with open('out_timestamps.txt', 'w') as timestampFile:
        timestampFile.write(f'{clipEnds}')
    logging.info(f'All timestamps saved to out_timestamps.txt')

    # Cut and export video according to clipEnds list
    logging.info(f'Cutting and exporting the final video...')
    cutAndExportVideo(VIDEO_FILENAME, VIDEO_OUT_FILENAME, clipEnds)
    logging.info(f'Video exported to {VIDEO_OUT_FILENAME}')
