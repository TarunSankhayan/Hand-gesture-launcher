# Hand-gesture-launcher
A Python-based utility script designed to allow users to control their computer and launch applications or URLs using real-time hand movements detected by a webcam.  It acts as a non-contact interface, translating specific gestures into system actions

Use real-time hand movements to launch applications, open files, and trigger URLs without touching your keyboard or mouse.

🌟 FeaturesReal-Time Gesture Recognition: Accurately detects directional swipes (swipe_up, swipe_down, etc.) and depth movements (push, pull).
Customizable Action Mapping: Easily define actions (URL, folder, command, shell) for each detected gesture.

Cross-Platform Compatibility: Actions are seamlessly executed on Windows, macOS, and Linux.Visual HUD: Provides instant feedback on hand direction, detected gesture, and operational status.🛠️ RequirementsThe project relies on computer vision and tracking libraries.

You will need Python and the following dependencies:
Bashpip install opencv-python mediapipe numpy

🚀 Getting Started1. 

1. Configure Actions Open the hand_gesture_launcher.py file and modify the ACTION_MAP dictionary (around line 30) to map your desired gestures to actions.
[[[[⚠️ Security Warning: Use caution when configuring cmd or shell actions. The script will execute any command defined here.]]]]

   EXAMPLE ACTION_MAP Customization:

   Example: Opens a URL in the default browser
   "swipe_right": - {"type": "url", "payload": "https://www.google.com"}, 
    
   Example: Opens the user's home directory (payload=None opens default home path)
   "push": - {"type": "folder", "payload": None}, 
    
   Example: Runs a command (e.g., opens a calculator on Windows)
   "swipe_up": - {"type": "cmd", "payload": ["calc.exe"]}, 
    
   Example: Opens a specific file path
   "pull": - {"type": "file", "payload": "C:\\Users\\Public\\Desktop\\MyDoc.pdf"},
   } 
  
3. Run the ScriptExecute the script from your terminal:
         Bashpython hand_gesture_launcher.py
A window titled "Gesture Launcher" will appear, showing your camera feed and the tracking interface.
---------------------------------------------------------------------------------------------------------------------------------------
📐 Technical Detection
The script uses MediaPipe Hands to analyze the hand's geometry in real-time.

1. Swipe Detection (X/Y Plane Movement)Swipe gestures are detected by tracking the velocity and displacement of the central hand landmark (index 9). A swipe registers only if the movement is significant (SWIPE_DISPLACEMENT) and consistently directional over a short time window (SWIPE_FRAMES).

2. Push/Pull Detection (Depth Movement)Depth movements are inferred by monitoring changes in the 2D bounding box area of the hand.

   Push: An area increase of more than PUSH_AREA_RATIO (e.g., 12%) over PUSH_SUSTAIN frames suggests the hand moved closer to the camera.
   Pull: An area decrease of less than 1.0 / PUSH_AREA_RATIO suggests the hand moved further away from the camera.
   
---------------------------------------------------------------------------------------------------------------------------------------

🔧 Configuration Parameters
Fine-tune the gesture sensitivity by adjusting these constants in hand_gesture_launcher.py:

Parameter       -                    Default     -         Description

TRACK_LANDMARK_IDX       -               9     -              The specific landmark index used for tracking motion (center of palm).

SMOOTHING_POS                        -   0.6     -            Factor for smoothing the tracked position to reduce noise.

SWIPE_DISPLACEMENT                  -    80   -               Minimum pixel distance for a motion to qualify as a swipe.

PUSH_AREA_RATIO                -         1.12     -           Required ratio for hand area change to trigger a push/pull gesture.

GESTURE_COOLDOWN_FRAMES               -  12         -         Frames to wait after an action is fired to prevent rapid, accidental re- triggering.

--------------------------------------------------------------------------------------------------------------------------------------

📄 License
This project is licensed under the MIT License.See the LICENSE file for full details.

--------------------------------------------------------------------------------------------------------------------------------------
🙏 Acknowledgements
OpenCV
MediaPipe Hands
