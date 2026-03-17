import pydirectinput
import pyautogui
import time
import keyboard
import os

# --- 1. SETTINGS & CALIBRATION ---
RETRY_BUTTON_POS = (816, 124)
CLOSE_BUTTON_POS = (968, 809)
BOSS_UI_TEXT_PIXEL = (868, 67)
PURE_WHITE = (254, 254, 254)
TRUNK_TEMPLATES = []

# --- NEW TRACKING VARIABLES ---
BOSSES_KILLED = 0
CRATES_PER_BOSS = 4
TIMEOUTS_OCCURRED = 0
TOTAL_RESETS = 0 
# ------------------------------

MAX_COMBAT_TIME = 20 * 60 
ARENA_TEMPLATES = [f'b{i}.png' for i in range(1, 5)]

PRECISION_CONFIDENCE = 0.80
TURN_90_PX = 400
TURN_180_PX = 800
# ---------------------------------

pydirectinput.FAILSAFE = True

def print_session_report():
    print("\n" + "="*30)
    print(f"📊 SESSION REPORT")
    print(f"Bosses Killed: {BOSSES_KILLED}")
    print(f"Est. Crates:  {BOSSES_KILLED * CRATES_PER_BOSS}")
    print(f"Combat Timeouts: {TIMEOUTS_OCCURRED}")
    print(f"Navigation Resets: {TOTAL_RESETS}")
    print("="*30 + "\n")

def check_kill_switch():
    if keyboard.is_pressed('k'):
        print("\n[!] Kill switch! Stopping...")
        print_session_report()
        pydirectinput.keyUp('w')
        exit()

def reset_character():
    print(">>> [RESET] Character reset sequence initiated (ESC -> R -> ENTER)...")
    set_shift_lock(False) 
    time.sleep(0.5)
    pydirectinput.press('esc')
    time.sleep(0.5)
    pydirectinput.press('r')
    time.sleep(0.5)
    pydirectinput.press('enter')
    print(">>> [RESET] Waiting for respawn...")
    time.sleep(6.0) 

def boss_is_alive():
    return pyautogui.pixelMatchesColor(BOSS_UI_TEXT_PIXEL[0], BOSS_UI_TEXT_PIXEL[1], PURE_WHITE, tolerance=25)

def set_shift_lock(want_active):
    print(f">>> Verifying Shift Lock (Target: {'ON' if want_active else 'OFF'})...")
    screen_w, screen_h = pyautogui.size()
    center_x = screen_w // 2
    test_x = center_x + 200
    
    pydirectinput.moveTo(test_x, screen_h // 2)
    time.sleep(0.1)
    
    curr_x, _ = pyautogui.position()
    dist = abs(curr_x - center_x)
    is_active = (dist < 50)
    
    if is_active == want_active:
        print("   -> State is correct.")
    else:
        print("   -> State incorrect. Toggling CTRL...")
        pydirectinput.press('ctrl')
        time.sleep(0.5)

def perfect_click(x, y):
    pydirectinput.moveTo(x, y)
    time.sleep(0.2)
    pydirectinput.moveRel(10, 0) 
    time.sleep(0.1)
    pydirectinput.moveRel(-10, 0)
    time.sleep(0.1)
    pydirectinput.mouseDown()
    time.sleep(0.15)
    pydirectinput.mouseUp()

def scan_and_orient():
    global TOTAL_RESETS
    print(f">>> Dual-Mode Scan (Target: {PRECISION_CONFIDENCE*100}%)...")
    set_shift_lock(True)
    time.sleep(0.5)
    
    screen_w, screen_h = pyautogui.size()
    third = screen_w / 3 
    found = False
    ALL_TEMPLATES = TRUNK_TEMPLATES + ARENA_TEMPLATES
    
    start_search_time = time.time()
    
    while not found:
        check_kill_switch()
        
        if time.time() - start_search_time > 20:
            print(">>> [!] No target found for 20s. Resetting...")
            TOTAL_RESETS += 1
            reset_character()
            start_search_time = time.time() 
            set_shift_lock(True) 
            continue

        time.sleep(0.7)
        match_location = None
        matched_template_name = ""
        search_region = (0, 0, screen_w, int(screen_h * 0.7))
        
        for template in ALL_TEMPLATES:
            if not os.path.exists(template): continue
            try:
                location = pyautogui.locateOnScreen(template, confidence=PRECISION_CONFIDENCE, region=search_region, grayscale=False)
                if location:
                    match_location = location
                    matched_template_name = template
                    print(f">>> MATCH FOUND: {template}")
                    break
            except: continue
        
        if match_location:
            obj_x = match_location.left + (match_location.width / 2)
            if matched_template_name in TRUNK_TEMPLATES:
                print(">>> Landmark: TRUNK. Snapping...")
                if obj_x < third: pydirectinput.moveRel(TURN_90_PX, 0, relative=True)
                elif obj_x > (third * 2): pydirectinput.moveRel(-TURN_90_PX, 0, relative=True)
                else: pydirectinput.moveRel(TURN_180_PX, 0, relative=True)
                time.sleep(0.5)
                found = True
            elif matched_template_name in ARENA_TEMPLATES:
                print(">>> Landmark: ARENA. Centering...")
                offset = obj_x - (screen_w / 2)
                if abs(offset) < 100:
                    print("   -> LOCKED ON.")
                    found = True
                else:
                    pydirectinput.moveRel(int(offset * 0.25), 0, relative=True)
                    time.sleep(0.8) 
        else:
            print(f">>> Searching... ({int(time.time() - start_search_time)}s)")
            pydirectinput.moveRel(300, 0, relative=True)

    # Charging sequence...
    pydirectinput.press('1') 
    time.sleep(0.2)
    pydirectinput.keyUp('w') 
    pydirectinput.keyDown('w')
    time.sleep(0.1)
    pydirectinput.press('e') 
    time.sleep(1.5) 
    pydirectinput.press('j') 
    time.sleep(2.0)

def combat_rotation():
    global BOSSES_KILLED, TIMEOUTS_OCCURRED
    print(">>> ENGAGING BOSS...")
    pydirectinput.keyDown('w')
    start_combat_time = time.time()
    
    while boss_is_alive():
        check_kill_switch()
        
        if (time.time() - start_combat_time) > MAX_COMBAT_TIME:
            print(f">>> [!] Combat Timeout ({MAX_COMBAT_TIME}s).")
            TIMEOUTS_OCCURRED += 1
            pydirectinput.keyUp('w')
            reset_character()
            print_session_report()
            return False 

        for skill in ['r', 'c', 'r', 'f', 'r']:
            if not boss_is_alive(): break
            pydirectinput.press(skill)
            time.sleep(0.4)
            
        if boss_is_alive():
            pydirectinput.press('2') 
            time.sleep(0.2)
            for skill in ['r', 'c', 'x']:
                if not boss_is_alive(): break
                pydirectinput.press(skill)
                time.sleep(0.4)
            pydirectinput.press('1') 

    pydirectinput.keyUp('w')
    time.sleep(0.5)
    BOSSES_KILLED += 1
    print(">>> Combat Complete.")
    print_session_report()
    set_shift_lock(False)
    return True

def retry_sequence():
    print(">>> Starting Retry Sequence...")
    time.sleep(4.0) 
    set_shift_lock(False)
    
    print(">>> Dismissing Rewards...")
    for i in range(3):
        perfect_click(CLOSE_BUTTON_POS[0], CLOSE_BUTTON_POS[1])
        time.sleep(0.8)

    print(">>> Clicking Retry...")
    for i in range(3):
        check_kill_switch()
        perfect_click(RETRY_BUTTON_POS[0], RETRY_BUTTON_POS[1])
        time.sleep(0.8)
    time.sleep(2.0)

if __name__ == "__main__":
    print("!!! HOLD 'K' TO KILL BOT !!!")
    time.sleep(5)
    try:
        while True:
            scan_and_orient()
            
            if boss_is_alive():
                success = combat_rotation()
                if success:
                    retry_sequence()
                    time.sleep(8)
                else:
                    time.sleep(2)
            else:
                pydirectinput.keyUp('w')
                set_shift_lock(False) 
                retry_sequence()
                time.sleep(8)
                
    except KeyboardInterrupt:
        pydirectinput.keyUp('w')