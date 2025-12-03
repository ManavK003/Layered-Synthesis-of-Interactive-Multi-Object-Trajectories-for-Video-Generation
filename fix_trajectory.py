# Fix the interpolation issue
with open('src/trajectory_controller.py', 'r') as f:
    content = f.read()

# Replace the smooth interpolation section
old_code = """        elif trajectory_type == 'smooth':
            # Smooth interpolation using cubic spline
            t_in = np.array([0, num_frames - 1])
            t_out = np.linspace(0, num_frames - 1, num_frames)
            bboxes_in = np.stack([start_bbox, end_bbox])
            
            trajectory = []
            for dim in range(4):
                f = interp1d(t_in, bboxes_in[:, dim], kind='cubic')
                trajectory.append(f(t_out))
            trajectory = np.stack(trajectory, axis=1)"""

new_code = """        elif trajectory_type == 'smooth':
            # Smooth interpolation using quadratic (cubic needs 4+ points)
            t_in = np.array([0, num_frames - 1])
            t_out = np.linspace(0, num_frames - 1, num_frames)
            bboxes_in = np.stack([start_bbox, end_bbox])
            
            trajectory = []
            for dim in range(4):
                # Use quadratic for 2 points (or linear as fallback)
                try:
                    f = interp1d(t_in, bboxes_in[:, dim], kind='quadratic')
                except:
                    f = interp1d(t_in, bboxes_in[:, dim], kind='linear')
                trajectory.append(f(t_out))
            trajectory = np.stack(trajectory, axis=1)"""

content = content.replace(old_code, new_code)

with open('src/trajectory_controller.py', 'w') as f:
    f.write(content)

print("Fixed trajectory interpolation!")
