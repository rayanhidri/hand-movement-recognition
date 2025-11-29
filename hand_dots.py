import cv2
import mediapipe as mp
import time
import math
import numpy as np
import random

mp_hands = mp.solutions.hands

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def detect_gesture(hand_landmarks):
    # Get finger tip positions
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Get palm base for reference
    wrist = hand_landmarks.landmark[0]
    
    # Check which fingers are extended (simple check based on y position)
    index_up = index_tip.y < hand_landmarks.landmark[6].y
    middle_up = middle_tip.y < hand_landmarks.landmark[10].y
    ring_up = ring_tip.y < hand_landmarks.landmark[14].y
    pinky_up = pinky_tip.y < hand_landmarks.landmark[18].y
    
    # Peace sign (index + middle up, others down)
    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
    
    # All fingers up
    elif index_up and middle_up and ring_up and pinky_up:
        return "open_hand"
    
    # Fist (no fingers up)
    elif not (index_up or middle_up or ring_up or pinky_up):
        return "fist"
    
    # Just index up
    elif index_up and not middle_up and not ring_up and not pinky_up:
        return "point"
    
    return "none"

class Particle:
    def __init__(self, x, y, vx=0, vy=0, color=(255, 200, 100), lifetime=30, size=3, particle_type="trail"):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size
        self.particle_type = particle_type
        self.angle = random.uniform(0, 2 * math.pi)
        self.rotation_speed = random.uniform(-0.2, 0.2)
    
    def update(self):
        self.lifetime -= 1
        
        # Physics based on particle type
        if self.particle_type == "firework":
            self.vy += 0.3  # Gravity
            self.x += self.vx
            self.y += self.vy
        elif self.particle_type == "star":
            self.angle += self.rotation_speed
            self.x += self.vx * 0.5
            self.y += self.vy * 0.5
        elif self.particle_type == "electric":
            self.x += self.vx + random.uniform(-2, 2)
            self.y += self.vy + random.uniform(-2, 2)
        else:
            self.x += self.vx
            self.y += self.vy
    
    def draw(self, frame):
        alpha = self.lifetime / self.max_lifetime
        
        if self.particle_type == "star":
            # Draw a rotating star
            points = []
            for i in range(5):
                angle = self.angle + i * 2 * math.pi / 5
                r = self.size * (1.5 if i % 2 == 0 else 0.7)
                px = int(self.x + r * math.cos(angle))
                py = int(self.y + r * math.sin(angle))
                points.append([px, py])
            
            color = tuple(int(c * alpha) for c in self.color)
            cv2.fillPoly(frame, [np.array(points)], color)
        
        elif self.particle_type == "electric":
            # Electric spark effect
            color = (int(100 + 155 * alpha), int(100 + 155 * alpha), 255)
            size = int(self.size * alpha)
            cv2.circle(frame, (int(self.x), int(self.y)), size, color, -1)
            
        else:
            # Regular particle with fade
            color = tuple(int(c * alpha) for c in self.color)
            size = int(self.size * alpha)
            cv2.circle(frame, (int(self.x), int(self.y)), size, color, -1)

class ExplosionEffect:
    def __init__(self, x, y):
        self.particles = []
        # Create explosion particles
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            color = random.choice([
                (255, 100, 100),  # Red
                (255, 200, 0),    # Orange
                (255, 255, 100),  # Yellow
            ])
            self.particles.append(
                Particle(x, y, vx, vy, color, lifetime=40, size=5, particle_type="firework")
            )
    
    def update(self):
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)
    
    def draw(self, frame):
        for p in self.particles:
            p.draw(frame)
    
    def is_alive(self):
        return len(self.particles) > 0

def apply_neon_glow(frame, particles):
    # Create a separate layer for glow
    glow = np.zeros_like(frame)
    
    for particle in particles:
        color = particle.color
        alpha = particle.lifetime / particle.max_lifetime
        
        # Draw multiple circles with decreasing alpha for glow effect
        for radius in range(particle.size * 3, particle.size, -1):
            glow_alpha = alpha * 0.1
            glow_color = tuple(int(c * glow_alpha) for c in color)
            cv2.circle(glow, (int(particle.x), int(particle.y)), radius, glow_color, -1)
    
    # Blur the glow
    glow = cv2.GaussianBlur(glow, (15, 15), 0)
    
    # Add glow to frame
    frame = cv2.add(frame, glow)
    return frame

def main():
    cap = None
    for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
        for idx in [0, 1]:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera opened")
                        break
                cap.release()
            except:
                pass
        if cap and cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    time.sleep(0.5)

    scale = 1.0
    prev_pinch_dist = None
    
    # Enhanced particle system
    particles = []
    explosions = []
    
    # Background effects
    background_pulse = 0
    screen_shake = 0
    
    # Previous positions for motion tracking
    prev_x, prev_y = None, None
    
    # Cooldown for gesture spawning
    gesture_cooldown = 0
    current_gesture = "none"

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        frame_count = 0
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape

            # Apply screen shake if active
            if screen_shake > 0:
                offset_x = int(random.uniform(-screen_shake, screen_shake))
                offset_y = int(random.uniform(-screen_shake, screen_shake))
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                frame = cv2.warpAffine(frame, M, (w, h))
                screen_shake *= 0.8

            # Background pulse effect
            background_pulse = (background_pulse + 0.05) % (2 * math.pi)
            pulse_intensity = int(20 * (math.sin(background_pulse) + 1) / 2)
            overlay = frame.copy()
            overlay[:, :] = (pulse_intensity, pulse_intensity // 2, pulse_intensity)
            frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            circle_x = w // 2
            circle_y = h // 2
            speed = 0

            # Decrease gesture cooldown
            if gesture_cooldown > 0:
                gesture_cooldown -= 1

            # Process hands
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand skeleton with neon effect
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks.landmark[start_idx]
                        end = hand_landmarks.landmark[end_idx]
                        start_pos = (int(start.x * w), int(start.y * h))
                        end_pos = (int(end.x * w), int(end.y * h))
                        
                        # Neon lines
                        cv2.line(frame, start_pos, end_pos, (100, 255, 255), 4)
                        cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
                    
                    # Draw landmarks with glow
                    for lm in hand_landmarks.landmark:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        cv2.circle(frame, (x_px, y_px), 8, (255, 100, 255), -1)
                        cv2.circle(frame, (x_px, y_px), 4, (255, 255, 255), -1)

                # First hand controls
                if len(results.multi_hand_landmarks) > 0:
                    hand = results.multi_hand_landmarks[0]
                    index_tip = hand.landmark[8]
                    thumb_tip = hand.landmark[4]
                    
                    circle_x = int(index_tip.x * w)
                    circle_y = int(index_tip.y * h)
                    
                    # Calculate speed for effects
                    if prev_x is not None:
                        dx = circle_x - prev_x
                        dy = circle_y - prev_y
                        speed = math.sqrt(dx**2 + dy**2)
                    
                    prev_x, prev_y = circle_x, circle_y
                    
                    # Detect gesture
                    detected_gesture = detect_gesture(hand)
                    
                    # Spawn particles based on gesture
                    if detected_gesture != "none" and gesture_cooldown == 0:
                        current_gesture = detected_gesture
                        gesture_cooldown = 10
                        
                        if detected_gesture == "peace":
                            # Spawn stars
                            for _ in range(5):
                                vx = random.uniform(-3, 3)
                                vy = random.uniform(-3, 3)
                                particles.append(
                                    Particle(circle_x, circle_y, vx, vy, 
                                           (255, 255, 100), lifetime=60, size=8, particle_type="star")
                                )
                        
                        elif detected_gesture == "fist":
                            # Explosion!
                            explosions.append(ExplosionEffect(circle_x, circle_y))
                            screen_shake = 15
                        
                        elif detected_gesture == "open_hand":
                            # Electric sparks
                            for _ in range(15):
                                angle = random.uniform(0, 2 * math.pi)
                                vx = math.cos(angle) * random.uniform(1, 4)
                                vy = math.sin(angle) * random.uniform(1, 4)
                                particles.append(
                                    Particle(circle_x, circle_y, vx, vy,
                                           (100, 200, 255), lifetime=30, size=4, particle_type="electric")
                                )
                    
                    # Regular trail particles (color based on speed)
                    trail_color = (
                        int(100 + min(speed * 10, 155)),
                        int(200 - min(speed * 5, 100)),
                        int(100 + min(speed * 5, 155))
                    )
                    particles.append(Particle(circle_x, circle_y, 0, 0, trail_color, lifetime=40, size=5))
                    
                    # Pinch to scale
                    pinch_dist = distance(index_tip, thumb_tip)
                    if prev_pinch_dist is not None:
                        delta = pinch_dist - prev_pinch_dist
                        scale += delta * 6
                        scale = max(0.3, min(5.0, scale))
                    prev_pinch_dist = pinch_dist
                
                # Two hands = electric connection
                if len(results.multi_hand_landmarks) == 2:
                    hand1 = results.multi_hand_landmarks[0]
                    hand2 = results.multi_hand_landmarks[1]
                    
                    index1 = hand1.landmark[8]
                    index2 = hand2.landmark[8]
                    
                    pos1 = (int(index1.x * w), int(index1.y * h))
                    pos2 = (int(index2.x * w), int(index2.y * h))
                    
                    # Draw electric arc between hands
                    num_segments = 8
                    for i in range(num_segments):
                        t = i / num_segments
                        mid_x = int(pos1[0] * (1 - t) + pos2[0] * t)
                        mid_y = int(pos1[1] * (1 - t) + pos2[1] * t)
                        
                        # Add random offset for electric effect
                        offset_x = random.randint(-15, 15)
                        offset_y = random.randint(-15, 15)
                        
                        end_x = int(pos1[0] * (1 - (t + 1/num_segments)) + pos2[0] * (t + 1/num_segments))
                        end_y = int(pos1[1] * (1 - (t + 1/num_segments)) + pos2[1] * (t + 1/num_segments))
                        
                        cv2.line(frame, 
                                (mid_x + offset_x, mid_y + offset_y),
                                (end_x + random.randint(-15, 15), end_y + random.randint(-15, 15)),
                                (200, 200, 255), 3)

            # Update and draw explosions
            for explosion in explosions[:]:
                explosion.update()
                explosion.draw(frame)
                if not explosion.is_alive():
                    explosions.remove(explosion)

            # Apply neon glow effect
            frame = apply_neon_glow(frame, particles)

            # Draw particles
            for particle in particles[:]:
                particle.update()
                particle.draw(frame)
                if particle.lifetime <= 0:
                    particles.remove(particle)
            
            # Draw main circle with extra effects
            size = int(60 * scale)
            
            # Outer glow rings
            for ring in range(3, 0, -1):
                ring_alpha = 0.3 * (ring / 3)
                ring_color = (
                    int(255 * ring_alpha),
                    int(200 * ring_alpha),
                    int(100 * ring_alpha)
                )
                cv2.circle(frame, (circle_x, circle_y), size + ring * 10, ring_color, 2)
            
            # Main circle
            cv2.circle(frame, (circle_x, circle_y), size, (255, 200, 100), 3)
            cv2.circle(frame, (circle_x, circle_y), size, (255, 255, 255), 1)
            
            # Rotating radial lines
            rotation = (frame_count * 0.05) % (2 * math.pi)
            for i in range(12):
                angle = rotation + i * math.pi / 6
                x = circle_x + int(size * math.cos(angle))
                y = circle_y + int(size * math.sin(angle))
                cv2.line(frame, (circle_x, circle_y), (x, y), (150, 150, 255), 2)
            
            # UI with gesture info
            ui_height = 100
            cv2.rectangle(frame, (10, 10), (300, ui_height), (0, 0, 0), -1)
            cv2.putText(frame, f"Scale: {scale:.2f}x", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {current_gesture}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            cv2.putText(frame, "Peace=Stars | Fist=Boom | Open=Electric", (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("HAND EFFECTS ENHANCED", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Done")

if __name__ == "__main__":
    main()