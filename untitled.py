#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on 五月 11, 2025, at 23:22
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'untitled'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\孔昊男\\Desktop\\ForR\\untitled.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('F_key_resp') is None:
        # initialise F_key_resp
        F_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='F_key_resp',
        )
    if deviceManager.getDevice('J_key_resp') is None:
        # initialise J_key_resp
        J_key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='J_key_resp',
        )
    if deviceManager.getDevice('rating') is None:
        # initialise rating
        rating = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='rating',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Exp_instruction" ---
    instruction = visual.TextStim(win=win, name='instruction',
        text='In the following task, a fixation point will appear at the center of the screen. After the fixation point disappears, a picture will be shown. This picture may be generated by AI technology or it may be a real photographed image. First, please make a judgment. If you think it is generated by AI(fake), press the F key. If you think it is a real picture(real), please press the J key. After making the judgment, please rate your confidence in your judgment. From 1 to 4, 1 indicates very low confidence and 4 indicates very high confidence. Press any key to start.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Exp_point" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon', vertices='cross',
        size=(0.2, 0.2),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "Exp_process" ---
    F_key = visual.TextStim(win=win, name='F_key',
        text='F: Fake',
        font='Arial',
        pos=(-0.5, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    J_key = visual.TextStim(win=win, name='J_key',
        text='Real : J',
        font='Arial',
        pos=(0.5, 0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    F_key_resp = keyboard.Keyboard(deviceName='F_key_resp')
    J_key_resp = keyboard.Keyboard(deviceName='J_key_resp')
    
    # --- Initialize components for Routine "Exp_conf" ---
    conf = visual.TextStim(win=win, name='conf',
        text='Please rate your confidence in the correctness of the judgment you just made.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rating = keyboard.Keyboard(deviceName='rating')
    
    # --- Initialize components for Routine "Exp_end" ---
    text = visual.TextStim(win=win, name='text',
        text='Thank you for participating in this experiment.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Exp_instruction" ---
    # create an object to store info about Routine Exp_instruction
    Exp_instruction = data.Routine(
        name='Exp_instruction',
        components=[instruction, key_resp],
    )
    Exp_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Exp_instruction
    Exp_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Exp_instruction.tStart = globalClock.getTime(format='float')
    Exp_instruction.status = STARTED
    thisExp.addData('Exp_instruction.started', Exp_instruction.tStart)
    Exp_instruction.maxDuration = None
    # keep track of which components have finished
    Exp_instructionComponents = Exp_instruction.components
    for thisComponent in Exp_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Exp_instruction" ---
    Exp_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction* updates
        
        # if instruction is starting this frame...
        if instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction.frameNStart = frameN  # exact frame index
            instruction.tStart = t  # local t and not account for scr refresh
            instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction.started')
            # update status
            instruction.status = STARTED
            instruction.setAutoDraw(True)
        
        # if instruction is active this frame...
        if instruction.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Exp_instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Exp_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Exp_instruction" ---
    for thisComponent in Exp_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Exp_instruction
    Exp_instruction.tStop = globalClock.getTime(format='float')
    Exp_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Exp_instruction.stopped', Exp_instruction.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Exp_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=2.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('images.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Exp_point" ---
        # create an object to store info about Routine Exp_point
        Exp_point = data.Routine(
            name='Exp_point',
            components=[polygon],
        )
        Exp_point.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Exp_point
        Exp_point.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exp_point.tStart = globalClock.getTime(format='float')
        Exp_point.status = STARTED
        thisExp.addData('Exp_point.started', Exp_point.tStart)
        Exp_point.maxDuration = None
        # keep track of which components have finished
        Exp_pointComponents = Exp_point.components
        for thisComponent in Exp_point.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Exp_point" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Exp_point.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.3:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon.started')
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon.stopped')
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Exp_point.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Exp_point.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exp_point" ---
        for thisComponent in Exp_point.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exp_point
        Exp_point.tStop = globalClock.getTime(format='float')
        Exp_point.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exp_point.stopped', Exp_point.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Exp_point.maxDurationReached:
            routineTimer.addTime(-Exp_point.maxDuration)
        elif Exp_point.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.300000)
        
        # --- Prepare to start Routine "Exp_process" ---
        # create an object to store info about Routine Exp_process
        Exp_process = data.Routine(
            name='Exp_process',
            components=[F_key, J_key, image, F_key_resp, J_key_resp],
        )
        Exp_process.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image.setImage(images)
        # create starting attributes for F_key_resp
        F_key_resp.keys = []
        F_key_resp.rt = []
        _F_key_resp_allKeys = []
        # create starting attributes for J_key_resp
        J_key_resp.keys = []
        J_key_resp.rt = []
        _J_key_resp_allKeys = []
        # store start times for Exp_process
        Exp_process.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exp_process.tStart = globalClock.getTime(format='float')
        Exp_process.status = STARTED
        thisExp.addData('Exp_process.started', Exp_process.tStart)
        Exp_process.maxDuration = None
        # keep track of which components have finished
        Exp_processComponents = Exp_process.components
        for thisComponent in Exp_process.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Exp_process" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Exp_process.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *F_key* updates
            
            # if F_key is starting this frame...
            if F_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                F_key.frameNStart = frameN  # exact frame index
                F_key.tStart = t  # local t and not account for scr refresh
                F_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(F_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'F_key.started')
                # update status
                F_key.status = STARTED
                F_key.setAutoDraw(True)
            
            # if F_key is active this frame...
            if F_key.status == STARTED:
                # update params
                pass
            
            # *J_key* updates
            
            # if J_key is starting this frame...
            if J_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                J_key.frameNStart = frameN  # exact frame index
                J_key.tStart = t  # local t and not account for scr refresh
                J_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'J_key.started')
                # update status
                J_key.status = STARTED
                J_key.setAutoDraw(True)
            
            # if J_key is active this frame...
            if J_key.status == STARTED:
                # update params
                pass
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # *F_key_resp* updates
            
            # if F_key_resp is starting this frame...
            if F_key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                F_key_resp.frameNStart = frameN  # exact frame index
                F_key_resp.tStart = t  # local t and not account for scr refresh
                F_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(F_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('F_key_resp.started', t)
                # update status
                F_key_resp.status = STARTED
                # keyboard checking is just starting
                F_key_resp.clock.reset()  # now t=0
            if F_key_resp.status == STARTED:
                theseKeys = F_key_resp.getKeys(keyList=['f'], ignoreKeys=["escape"], waitRelease=False)
                _F_key_resp_allKeys.extend(theseKeys)
                if len(_F_key_resp_allKeys):
                    F_key_resp.keys = _F_key_resp_allKeys[-1].name  # just the last key pressed
                    F_key_resp.rt = _F_key_resp_allKeys[-1].rt
                    F_key_resp.duration = _F_key_resp_allKeys[-1].duration
                    # was this correct?
                    if (F_key_resp.keys == str(judge)) or (F_key_resp.keys == judge):
                        F_key_resp.corr = 1
                    else:
                        F_key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *J_key_resp* updates
            
            # if J_key_resp is starting this frame...
            if J_key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                J_key_resp.frameNStart = frameN  # exact frame index
                J_key_resp.tStart = t  # local t and not account for scr refresh
                J_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(J_key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('J_key_resp.started', t)
                # update status
                J_key_resp.status = STARTED
                # keyboard checking is just starting
                J_key_resp.clock.reset()  # now t=0
            if J_key_resp.status == STARTED:
                theseKeys = J_key_resp.getKeys(keyList=['j'], ignoreKeys=["escape"], waitRelease=False)
                _J_key_resp_allKeys.extend(theseKeys)
                if len(_J_key_resp_allKeys):
                    J_key_resp.keys = _J_key_resp_allKeys[-1].name  # just the last key pressed
                    J_key_resp.rt = _J_key_resp_allKeys[-1].rt
                    J_key_resp.duration = _J_key_resp_allKeys[-1].duration
                    # was this correct?
                    if (J_key_resp.keys == str(judge)) or (J_key_resp.keys == judge):
                        J_key_resp.corr = 1
                    else:
                        J_key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from code
            keys = event.waitKeys(keyList=['escape'])
            if 'escape' in keys:
                core.quit()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Exp_process.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Exp_process.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exp_process" ---
        for thisComponent in Exp_process.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exp_process
        Exp_process.tStop = globalClock.getTime(format='float')
        Exp_process.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exp_process.stopped', Exp_process.tStop)
        # check responses
        if F_key_resp.keys in ['', [], None]:  # No response was made
            F_key_resp.keys = None
            # was no response the correct answer?!
            if str(judge).lower() == 'none':
               F_key_resp.corr = 1;  # correct non-response
            else:
               F_key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('F_key_resp.keys',F_key_resp.keys)
        trials.addData('F_key_resp.corr', F_key_resp.corr)
        if F_key_resp.keys != None:  # we had a response
            trials.addData('F_key_resp.rt', F_key_resp.rt)
            trials.addData('F_key_resp.duration', F_key_resp.duration)
        # check responses
        if J_key_resp.keys in ['', [], None]:  # No response was made
            J_key_resp.keys = None
            # was no response the correct answer?!
            if str(judge).lower() == 'none':
               J_key_resp.corr = 1;  # correct non-response
            else:
               J_key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('J_key_resp.keys',J_key_resp.keys)
        trials.addData('J_key_resp.corr', J_key_resp.corr)
        if J_key_resp.keys != None:  # we had a response
            trials.addData('J_key_resp.rt', J_key_resp.rt)
            trials.addData('J_key_resp.duration', J_key_resp.duration)
        # the Routine "Exp_process" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Exp_conf" ---
        # create an object to store info about Routine Exp_conf
        Exp_conf = data.Routine(
            name='Exp_conf',
            components=[conf, rating],
        )
        Exp_conf.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for rating
        rating.keys = []
        rating.rt = []
        _rating_allKeys = []
        # store start times for Exp_conf
        Exp_conf.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exp_conf.tStart = globalClock.getTime(format='float')
        Exp_conf.status = STARTED
        thisExp.addData('Exp_conf.started', Exp_conf.tStart)
        Exp_conf.maxDuration = None
        # keep track of which components have finished
        Exp_confComponents = Exp_conf.components
        for thisComponent in Exp_conf.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Exp_conf" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Exp_conf.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *conf* updates
            
            # if conf is starting this frame...
            if conf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                conf.frameNStart = frameN  # exact frame index
                conf.tStart = t  # local t and not account for scr refresh
                conf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(conf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'conf.started')
                # update status
                conf.status = STARTED
                conf.setAutoDraw(True)
            
            # if conf is active this frame...
            if conf.status == STARTED:
                # update params
                pass
            
            # *rating* updates
            
            # if rating is starting this frame...
            if rating.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rating.frameNStart = frameN  # exact frame index
                rating.tStart = t  # local t and not account for scr refresh
                rating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('rating.started', t)
                # update status
                rating.status = STARTED
                # keyboard checking is just starting
                rating.clock.reset()  # now t=0
            if rating.status == STARTED:
                theseKeys = rating.getKeys(keyList=['1','2','3','4','5'], ignoreKeys=["escape"], waitRelease=False)
                _rating_allKeys.extend(theseKeys)
                if len(_rating_allKeys):
                    rating.keys = _rating_allKeys[-1].name  # just the last key pressed
                    rating.rt = _rating_allKeys[-1].rt
                    rating.duration = _rating_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Exp_conf.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Exp_conf.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exp_conf" ---
        for thisComponent in Exp_conf.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exp_conf
        Exp_conf.tStop = globalClock.getTime(format='float')
        Exp_conf.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exp_conf.stopped', Exp_conf.tStop)
        # check responses
        if rating.keys in ['', [], None]:  # No response was made
            rating.keys = None
        trials.addData('rating.keys',rating.keys)
        if rating.keys != None:  # we had a response
            trials.addData('rating.rt', rating.rt)
            trials.addData('rating.duration', rating.duration)
        # the Routine "Exp_conf" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 2.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Exp_end" ---
    # create an object to store info about Routine Exp_end
    Exp_end = data.Routine(
        name='Exp_end',
        components=[text, key_resp_2],
    )
    Exp_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for Exp_end
    Exp_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Exp_end.tStart = globalClock.getTime(format='float')
    Exp_end.status = STARTED
    thisExp.addData('Exp_end.started', Exp_end.tStart)
    Exp_end.maxDuration = None
    # keep track of which components have finished
    Exp_endComponents = Exp_end.components
    for thisComponent in Exp_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Exp_end" ---
    Exp_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('key_resp_2.started', t)
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            key_resp_2.clock.reset()  # now t=0
        if key_resp_2.status == STARTED:
            theseKeys = key_resp_2.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Exp_end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Exp_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Exp_end" ---
    for thisComponent in Exp_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Exp_end
    Exp_end.tStop = globalClock.getTime(format='float')
    Exp_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Exp_end.stopped', Exp_end.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "Exp_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
