# UAV_Indoor_ArUco_Landing-
# UAV Indoor ArUco Landing System

## Overview

This repository contains the **final working Python scripts** used for an indoor UAV precision landing system based on ArUco marker detection.

The project was developed as part of an aerospace engineering capstone, focusing on **vision-based navigation in a GPS-denied environment**.

⚠️ Note:  
This repository only contains the **core Python scripts that worked in the final system**.  
It does **not include the full ROS2 workspace or development history**.

---

## System Description

The system enables a UAV to:

- Perform a search pattern in an indoor environment  
- Detect an ArUco marker using a camera  
- Estimate relative position error  
- Apply velocity-based corrections  
- Align with the target  
- Execute a controlled landing  

The control approach is based on a **closed-loop feedback system**, where position errors are continuously corrected using velocity commands.

---

## Software Stack

The system was built using the following technologies:

- **ROS 2 (Humble)**
- **MAVROS**
- **MAVLink (via UDP)**
- **OpenCV (ArUco detection)**
- **Python 3**

---

## System Architecture (High-Level)
- Camera publishes image stream  
- ArUco detection computes marker pose  
- Error is converted into velocity commands (Vx, Vy)  
- Commands are sent via MAVROS to the flight controller  

---

## Requirements

To run these scripts, the following must already be configured:

### 1. ROS 2
- ROS 2 Humble installed
- Workspace properly sourced

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
