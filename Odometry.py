import cv2
import numpy as np
import time
import plotly.graph_objects as go
import csv
import pandas as pd
import plotly.express as px

def smooth_positions_adaptive_with_prediction(positions, alpha, speed_threshold, lookahead=1):
    smoothed = []
    prev = positions[0]
    smoothed.append(prev)

    for i in range(1, len(positions)):
        current = positions[i]
        prev_smooth = smoothed[-1]

        if i + lookahead < len(positions):
            future = positions[i + lookahead]
            predicted = tuple(
                current[j] + (future[j] - current[j]) for j in range(len(current))
            )
        else:
            predicted = current

        speed = np.linalg.norm(np.array(predicted) - np.array(prev_smooth))

        if speed > speed_threshold:
            new = tuple(alpha * predicted[j] + (1 - alpha) * prev_smooth[j] for j in range(len(current)))
        else:
            new = current

        smoothed.append(new)

    return smoothed

def nothing(x):
    pass

tracker = None
tracking = False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

cv2.namedWindow("Drone Head Tracking with Smoothing")
cv2.createTrackbar("Alpha x100", "Drone Head Tracking with Smoothing", 30, 100, nothing) 
cv2.createTrackbar("Speed Threshold", "Drone Head Tracking with Smoothing", 20, 200, nothing) 
start_time = time.time()
duration = 15

positions = []
kalman_positions = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

kalman = cv2.KalmanFilter(6, 3)
kalman.measurementMatrix = np.eye(3, 6, dtype=np.float32)
kalman.transitionMatrix = np.array([
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
], dtype=np.float32)
kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_display = frame.copy()

    if not tracking:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            tracker = cv2.TrackerCSRT_create()

            tracker.init(frame, (x, y, w, h)) 
            tracking = True
            cv2.putText(frame_display, "Tracking started", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        else:
            cv2.putText(frame_display, "No head detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
    else:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            center_x = x + w // 2
            center_y = y + h // 2
            z = 10000 / (w * h + 1)
            measurement = np.array([[np.float32(center_x)],
                        [np.float32(center_y)],
                        [np.float32(z)]])

            if len(positions) == 1:
                kalman.statePre = np.array([[center_x], [center_y], [z], [0], [0], [0]], dtype=np.float32)
                kalman.statePost = kalman.statePre.copy()

            prediction = kalman.predict()
            estimated = kalman.correct(measurement)

            kalman_position = (estimated[0, 0], estimated[1, 0], estimated[2, 0])

            positions.append((center_x, center_y, z))
            kalman_positions.append(kalman_position)
            cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_display, "Tracking head", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        else:
            cv2.putText(frame_display, "Lost track, searching...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
            tracking = False

    cv2.imshow("Drone Head Tracking with Smoothing", frame_display)

    alpha_slider = cv2.getTrackbarPos("Alpha x100", "Drone Head Tracking with Smoothing")
    speed_threshold = cv2.getTrackbarPos("Speed Threshold", "Drone Head Tracking with Smoothing")

    alpha = max(alpha_slider / 100.0, 0.01) 

    if cv2.waitKey(30) & 0xFF == 27:
        break

    if time.time() - start_time > duration:
        break

cap.release()
cv2.destroyAllWindows()

if positions:
    smoothed_positions = smooth_positions_adaptive_with_prediction(positions, alpha, speed_threshold, lookahead=1)

    xs, ys, zs = zip(*positions)
    xs_s, ys_s, zs_s = zip(*smoothed_positions)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines+markers',
        name='Оригінал',
        line=dict(color='red'),
        marker=dict(size=3)
    ))

    fig.add_trace(go.Scatter3d(
        x=xs_s, y=ys_s, z=zs_s,
        mode='lines+markers',
        name='Згладжена',
        line=dict(color='blue'),
        marker=dict(size=3)
    ))

    xk, yk, zk = zip(*kalman_positions)
    fig.add_trace(go.Scatter3d(
        x=xk, y=yk, z=zk,
        mode='lines+markers',
        name='Kalman-фільтр',
        line=dict(color='green'),
        marker=dict(size=3)
    ))

    fig.update_layout(
        title="Порівняння руху дрона (голова) з адаптивною фільтрацією різких рухів",
        scene=dict(
            xaxis_title='X (пікселі)',
            yaxis_title='Y (пікселі)',
            zaxis_title='Z (умовна відстань)'
        )
    )

    fig.show()

else:
    print("Трекінг не виконувався або не було знайдено голову")
import csv

def export_to_csv(filename, positions, smoothed_positions, kalman_positions):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame", 
            "x_raw", "y_raw", "z_raw", 
            "x_smooth", "y_smooth", "z_smooth", 
            "x_kalman", "y_kalman", "z_kalman"
        ])
        for i in range(len(positions)):
            row = [
            i,
            *positions[i],
            *(smoothed_positions[i] if i < len(smoothed_positions) else ("", "", "")),
            *(kalman_positions[i] if i < len(kalman_positions) else ("", "", ""))
            ]
            writer.writerow(row)

export_to_csv("drone_head_tracking_data.csv", positions, smoothed_positions, kalman_positions)
print("Дані експортовано у drone_head_tracking_data.csv")

drone_size = 5

def drone_model_points(x, y, z):
    return [
        (x, y, z),
        (x + drone_size, y, z),
        (x - drone_size / 2, y + drone_size * 0.866, z),
        (x, y, z)  
    ]

frames = []
n = len(xs)

for i in range(n):
    orig_points = drone_model_points(xs[i], ys[i], zs[i])
    smooth_points = drone_model_points(xs_s[i], ys_s[i], zs_s[i])

    frame = go.Frame(
        data=[
            # Оригінал: лінія + точки
            go.Scatter3d(x=xs[:i+1], y=ys[:i+1], z=zs[:i+1],
                         mode='lines+markers', line=dict(color='red'), marker=dict(size=4), name='Оригінал'),
            # Модель дрона оригінал (трикутник)
            go.Scatter3d(x=[p[0] for p in orig_points], y=[p[1] for p in orig_points], z=[p[2] for p in orig_points],
                         mode='lines+markers', line=dict(color='red'), marker=dict(size=6), name='Дрон (оригінал)'),

            # Згладжена: лінія + точки
            go.Scatter3d(x=xs_s[:i+1], y=ys_s[:i+1], z=zs_s[:i+1],
                         mode='lines+markers', line=dict(color='blue'), marker=dict(size=4), name='Згладжена'),
            # Модель дрона згладжена (трикутник)
            go.Scatter3d(x=[p[0] for p in smooth_points], y=[p[1] for p in smooth_points], z=[p[2] for p in smooth_points],
                         mode='lines+markers', line=dict(color='blue'), marker=dict(size=6), name='Дрон (згладжений)'),
        ],
        name=str(i)
    )
    frames.append(frame)

orig_points_0 = drone_model_points(xs[0], ys[0], zs[0])
smooth_points_0 = drone_model_points(xs_s[0], ys_s[0], zs_s[0])

fig = go.Figure(
    data=[
        go.Scatter3d(x=xs[:1], y=ys[:1], z=zs[:1],
                     mode='lines+markers', line=dict(color='red'), marker=dict(size=4), name='Оригінал'),
        go.Scatter3d(x=[p[0] for p in orig_points_0], y=[p[1] for p in orig_points_0], z=[p[2] for p in orig_points_0],
                     mode='lines+markers', line=dict(color='red'), marker=dict(size=6), name='Дрон (оригінал)'),

        go.Scatter3d(x=xs_s[:1], y=ys_s[:1], z=zs_s[:1],
                     mode='lines+markers', line=dict(color='blue'), marker=dict(size=4), name='Згладжена'),
        go.Scatter3d(x=[p[0] for p in smooth_points_0], y=[p[1] for p in smooth_points_0], z=[p[2] for p in smooth_points_0],
                     mode='lines+markers', line=dict(color='blue'), marker=dict(size=6), name='Дрон (згладжений)'),
    ],
    
    layout=go.Layout(
        title="Анімація руху дрона по оригінальній і згладженій траєкторіях",
        scene=dict(
            xaxis_title='X (пікселі)',
            yaxis_title='Y (пікселі)',
            zaxis_title='Z (умовна відстань)'
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1,
            x=1.1,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
            ],
        )],
    ),
    frames=frames
)

fig.show()
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    y=[p[0] for p in positions],
    mode='lines+markers',
    name='Raw X'
))

fig2.add_trace(go.Scatter(
    y=[p[0] for p in smoothed_positions],
    mode='lines+markers',
    name='Smoothed X'
))

fig2.add_trace(go.Scatter(
    y=[p[0] for p in kalman_positions],
    mode='lines+markers',
    name='Kalman X'
))

fig2.update_layout(title="Порівняння X координат по кадрах", xaxis_title="Кадр", yaxis_title="X (пікселі)")

fig2.show()

def load_positions_from_csv(filename):
    positions = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x = float(row['x_raw'])
            y = float(row['y_raw'])
            z = float(row['z_raw'])
            positions.append((x, y, z))
    return positions

def exponential_smoothing(data, alpha):
    """Експоненційне згладжування для списку чисел"""
    smoothed = []
    s = data[0]  
    smoothed.append(s)
    for i in range(1, len(data)):
        s = alpha * data[i] + (1 - alpha) * s
        smoothed.append(s)
    return smoothed

def smooth_positions(positions, alpha):
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    zs = [pos[2] for pos in positions]

    xs_smooth = exponential_smoothing(xs, alpha)
    ys_smooth = exponential_smoothing(ys, alpha)
    zs_smooth = exponential_smoothing(zs, alpha)

    return list(zip(xs_smooth, ys_smooth, zs_smooth))

def export_to_csv(filename, positions, smoothed_positions):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x_raw", "y_raw", "z_raw", "x_smooth", "y_smooth", "z_smooth"])
        for i in range(len(positions)):
            writer.writerow([
                i,
                *positions[i],
                *smoothed_positions[i]
            ])

if __name__ == "__main__":
    input_csv = "/Users/artem/Docs/Life/HomeWork/III Course/ЗбірДаних/projectOdometry/drone_head_tracking_data.csv"
    output_csv = "positions_smoothed.csv"

    alpha = 0.3
    positions = load_positions_from_csv(input_csv)
    smoothed_positions = smooth_positions(positions, alpha)
    export_to_csv(output_csv, positions, smoothed_positions)
    print(f"Згладжені позиції збережені у {output_csv}")

    def load_positions_from_csv(filename):
        positions = []
        smoothed_positions = []
        kalman_positions = []

        with open(filename, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                positions.append((
                    float(row['x_raw']),
                    float(row['y_raw']),
                    float(row['z_raw'])
                ))
                smoothed_positions.append((
                    float(row['x_smooth']),
                    float(row['y_smooth']),
                    float(row['z_smooth'])
                ))
                kalman_positions.append((
                    float(row['x_kalman']),
                    float(row['y_kalman']),
                    float(row['z_kalman'])
                ))
        return positions, smoothed_positions, kalman_positions

    def calculate_features(positions):
        features = []
        positions_np = np.array(positions)
        velocities = np.diff(positions_np, axis=0, prepend=positions_np[0:1])
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])

        for i in range(len(positions)):
            speed = np.linalg.norm(velocities[i])
            acceleration = np.linalg.norm(accelerations[i])
            features.append({
                'frame': i,
                'speed': speed,
                'acceleration': acceleration,
                'position': positions[i]
            })
        return features
    
    filename = 'drone_head_tracking_data.csv'
    positions, smoothed_positions, kalman_positions = load_positions_from_csv(filename)
    features = calculate_features(positions)

    for f in features[:5]:
        print(f)
df = pd.read_csv('positions_smoothed.csv')

speeds = [0.0]
for i in range(1, len(df)):
    dx = df.loc[i, 'x_smooth'] - df.loc[i - 1, 'x_smooth']
    dy = df.loc[i, 'y_smooth'] - df.loc[i - 1, 'y_smooth']
    dz = df.loc[i, 'z_smooth'] - df.loc[i - 1, 'z_smooth']
    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    speeds.append(speed)
df['speed'] = speeds

fig = px.scatter_3d(
    df,
    x='x_smooth',
    y='y_smooth',
    z='z_smooth',
    color='speed',
    size_max=5,
    title='3D Траєкторія руху з кольором за швидкістю',
    labels={'speed': 'Швидкість'},
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(scene=dict(
    xaxis_title='X',
    yaxis_title='Y',
    zaxis_title='Z'
))
fig.show()

def smooth_series(series, alpha=0.3):
    smoothed = [series[0]]
    for i in range(1, len(series)):
        smoothed.append(alpha * series[i] + (1 - alpha) * smoothed[i - 1])
    return smoothed
df = pd.read_csv('positions_smoothed.csv')

speeds = [0.0]
for i in range(1, len(df)):
    dx = df.loc[i, 'x_smooth'] - df.loc[i - 1, 'x_smooth']
    dy = df.loc[i, 'y_smooth'] - df.loc[i - 1, 'y_smooth']
    dz = df.loc[i, 'z_smooth'] - df.loc[i - 1, 'z_smooth']
    speed = np.sqrt(dx**2 + dy**2 + dz**2)
    speeds.append(speed)
df['speed'] = speeds

accelerations = [0.0]
for i in range(1, len(df)):
    acceleration = abs(df.loc[i, 'speed'] - df.loc[i - 1, 'speed'])
    accelerations.append(acceleration)
df['acceleration'] = accelerations

df['speed_smoothed'] = smooth_series(df['speed'], alpha=0.3)
df['acceleration_smoothed'] = smooth_series(df['acceleration'], alpha=0.3)

fig = go.Figure()

# Швидкість
fig.add_trace(go.Scatter(
    x=df['frame'], y=df['speed'],
    mode='lines+markers', name='Швидкість (оригінал)',
    line=dict(color='blue', width=1)
))
fig.add_trace(go.Scatter(
    x=df['frame'], y=df['speed_smoothed'],
    mode='lines', name='Швидкість (згладжена)',
    line=dict(color='blue', width=3, dash='dash')
))

# Прискорення
fig.add_trace(go.Scatter(
    x=df['frame'], y=df['acceleration'],
    mode='lines+markers', name='Прискорення (оригінал)',
    line=dict(color='red', width=1)
))
fig.add_trace(go.Scatter(
    x=df['frame'], y=df['acceleration_smoothed'],
    mode='lines', name='Прискорення (згладжене)',
    line=dict(color='red', width=3, dash='dash')
))

# Налаштування
fig.update_layout(
    title='Швидкість та прискорення (згладжені)',
    xaxis_title='Кадр',
    yaxis_title='Значення (пікселі/кадр)',
    legend=dict(x=0.01, y=0.99),
    template='plotly_white'
)

fig.show()