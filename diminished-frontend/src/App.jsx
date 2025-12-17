import React, { useRef, useEffect, useState, useCallback } from "react";
import { GoGear } from "react-icons/go";
import { MdFullscreen, MdFullscreenExit } from "react-icons/md";
import Webcam from "react-webcam";
import Settings from "./Settings";
import { diminishObject } from './DiminishObject';
import { isIOS } from 'react-device-detect';
import "./App.css";

const socket = new WebSocket('wss://diminish.soaratorium.com:5174');

function App() {
  const WIDTH = 640;
  const HEIGHT = 640;

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMovingTooFast, setIsMovingTooFast] = useState(false);
  const motionTimeoutRef = useRef(null);

  // For RTT testing.
  const pendingRequests = useRef(new Map());
  const rttArray = useRef([]);
  const MAX_RTT_RECORDS = 1000; 

  // Settings values
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [diminishEffect, setDiminishEffect] = useState(1);
  const [useOutline, setUseOutline] = useState(0); // 0 = Off, 1 = Healthy, 2 = All
  const [outlineColor, setOutlineColor] = useState('health_based');
  const [classOverrides, setClassOverrides] = useState({});

  const settingsRef = useRef({
    diminishEffect,
    useOutline,
    outlineColor,
    classOverrides
  });

  const detect = useCallback(async () => {
    if (!webcamRef.current || socket.readyState !== WebSocket.OPEN) return;

    try {
      const imageSrc = webcamRef.current.getScreenshot({ width: WIDTH, height: HEIGHT });
      if (!imageSrc) return;

      const requestId = Date.now() + Math.random();
      const startTime = performance.now();
      pendingRequests.current.set(requestId, startTime);

      // Send the screenshot through WebSocket
      socket.send(JSON.stringify({
        image: imageSrc,
        requestId: requestId
      }));

    } catch (error) {
      console.error('Detection error:', error);
    }
  }, []);

  // Toggle fullscreen
  const toggleFullscreen = useCallback(() => {
    // Iphones fullscreen
    if (isIOS) {
      if (containerRef.current.webkitRequestFullscreen) {
        containerRef.current.webkitRequestFullscreen();
        setIsFullscreen(true);
      }
      else {
        document.webkitExitFullscreen();
        setIsFullscreen(false);
      }
    }
    // Base fullscreen
    else {
      if (!document.fullscreenElement) {
        containerRef.current.requestFullscreen()
          .then(() => setIsFullscreen(true))
      } 
      else {
        document.exitFullscreen()
          .then(() => setIsFullscreen(false));
      }
    }
  }, []);

  const handleResize = () => {
    const video = webcamRef.current.video;
    const canvas = canvasRef.current;
    
    if (video && canvas) {
      const videoRect = video.getBoundingClientRect();
      canvas.width = videoRect.width;
      canvas.height = videoRect.height;
    }
  };

  // Initialize WebSocket connection
  useEffect(() => {
    socket.onopen = () => {
      console.log('WebSocket connection established.');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);

      console.log("Message received from server:", data);

      if (data.msg) {
        console.log('Server status:', data.msg);
        return;
      }

      const endTime = performance.now();
      const requestId = data.requestId;

      // Calculate RTT if we have the start time
      if (requestId && pendingRequests.current.has(requestId)) {
        const startTime = pendingRequests.current.get(requestId);
        const RTT = endTime - startTime;
        console.log(RTT);
        pendingRequests.current.delete(requestId);

        if (rttArray.current.length < MAX_RTT_RECORDS) {
          console.log("push")
          rttArray.current.push(RTT);
        }
        else {
          console.log(rttArray);
        }
      }

      // Process the detection results
      if (canvasRef.current && webcamRef.current?.video) {
        diminishObject(
          canvasRef.current,
          webcamRef.current.video,
          data.detections,
          settingsRef.current.diminishEffect,
          settingsRef.current.useOutline,
          settingsRef.current.outlineColor,
          settingsRef.current.classOverrides
        );
      }
    };

    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    socket.onclose = (event) => {
      console.log('WebSocket connection closed.:', error);
    };

    return () => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    };
  }, []);

  // Handle resize to keep canvas aligned with video
  useEffect(() => {
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  // Handle fullscreen.
  useEffect(() => {
    const fullscreenHandler = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', fullscreenHandler);
    return () => document.removeEventListener('fullscreenchange', fullscreenHandler);
  }, []);

  // Motion detection
  useEffect(() => {
    const setupMotionDetection = () => {
      if (window.DeviceMotionEvent) {
        const handleMotionEvent = (event) => {
          const { accelerationIncludingGravity, rotationRate } = event;
          
          const accel = accelerationIncludingGravity || { x: 0, y: 0, z: 0 };
          const rotRate = rotationRate || { alpha: 0, beta: 0, gamma: 0 };

          const accelerationMagnitude = Math.sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
          const rotationMagnitude = Math.sqrt(rotRate.alpha * rotRate.alpha + rotRate.beta * rotRate.beta + rotRate.gamma * rotRate.gamma);

          const ACCEL_THRESHOLD = 20; // m/s^2
          const ROTATION_THRESHOLD = 270; // deg/s

          if (accelerationMagnitude > ACCEL_THRESHOLD || rotationMagnitude > ROTATION_THRESHOLD) {
            setIsMovingTooFast(true);

            if (motionTimeoutRef.current) clearTimeout(motionTimeoutRef.current);
            
            motionTimeoutRef.current = setTimeout(() => {
              setIsMovingTooFast(false);
            }, 1000);
          }
        };
        
        if (typeof DeviceMotionEvent.requestPermission === 'function') {
          DeviceMotionEvent.requestPermission()
            .then(permissionState => {
              if (permissionState === 'granted') {
                window.addEventListener('devicemotion', handleMotionEvent);
              }
            })
            .catch(console.error);
        } else {
          window.addEventListener('devicemotion', handleMotionEvent);
        }

        return () => window.removeEventListener('devicemotion', handleMotionEvent);
      } else {
        console.log("Device motion not supported on this device.");
      }
    };

    const cleanupMotion = setupMotionDetection();

    return () => {
      if (cleanupMotion) {
        cleanupMotion();
      }
      if (motionTimeoutRef.current) {
        clearTimeout(motionTimeoutRef.current);
      }
    };
  }, []);

  // Detect the objects on the screen every interval.
  useEffect(() => {
    const detectionInterval = setInterval(detect, 200);
    return () => clearInterval(detectionInterval);
  }, [detect]);

  // Update the ref when settings change.
  useEffect(() => {
    settingsRef.current = {
      diminishEffect,
      useOutline,
      outlineColor,
      classOverrides
    };
  }, [diminishEffect, useOutline, outlineColor, classOverrides]);

  return (
    <div className={`container ${isMovingTooFast ? 'moving-too-fast' : ''}`} ref={containerRef}>
      <div className="header">
        <GoGear className="gear-icon" onClick={() => setSettingsOpen(!settingsOpen)}/>
        {isFullscreen ? (
          <MdFullscreenExit className="control-icon" onClick={toggleFullscreen} />
        ) : (
          <MdFullscreen className="control-icon" onClick={toggleFullscreen} />
        )}
      </div>

      <Webcam
        className="webcam-component"
        ref={webcamRef}
        onLoadedMetadata={handleResize}
        width={WIDTH}
        height={HEIGHT}
        screenshotFormat="image/jpeg"
        videoConstraints={{
          facingMode: "environment",
        }}
      />
      <canvas className="canvas-overlay" ref={canvasRef}/>

      <Settings
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        diminishEffect={diminishEffect}
        setDiminishEffect={setDiminishEffect}
        useOutline={useOutline}
        setUseOutline={setUseOutline}
        outlineColor={outlineColor}
        setOutlineColor={setOutlineColor}
        classOverrides={classOverrides}
        setClassOverrides={setClassOverrides}
      />
    </div>
  );
}

export default App;