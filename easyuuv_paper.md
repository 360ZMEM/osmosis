# EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control

Guanwen Xie1,†, Student Member, IEEE, Jingzehua Xu2,†,B, Student Member, IEEE, Jiwei Tang3, Student Member, IEEE, Yubo Huang4, Zixi Wang5, Shuai Zhang6, Member, IEEE, Dongfang Ma7, Member, IEEE, Juntian Qu1, Member, IEEE, and Xiaofan Li2 

Abstract—Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. To facilitate reproducibility and further research, the source code, LLM prompts, and supplementary video are provided in the following repositories: 

§ Homepage: https://360zmem.github.io/easyuuv/ 

Å Video: https://youtu.be/m2yLQzxiILc 

Index Terms—Unmanned Underwater Vehicle, Reinforcement Learning, Large Language Model, Simulation to Reality, Attitude Control 

1G. Xie and J. Qu are with Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, 518055, China. E-mail: xgw24@mails.tsinghua.edu.cn, juntian.qu@sz.tsinghua.edu.cn. 

2J. Xu and X. Li are with Department of Mechanical Engineering, The University of Hong Kong, Pokfulam Road, Hong Kong, China; E-mail: xjzh23@mails.tsinghua.edu.cn, lixf@hku.hk. 

3J. Tang is with Department of Data and Systems Engineering, The University of Hong Kong, Pokfulam Road, Hong Kong, China; Email: tangjiwei@connect.hku.hk. 

4Y. Huang is with School of Civil Engineering, Southwest Jiaotong University, Chengdu, 611756, China; E-mail: ybforever@my.swjtu.edu.cn. 

5Z. Wang is with School of Information and Software Engineering, University of Electronic Science and Technology of China, Chengdu, 611731, China; E-mail: 202521090118@std.uestc.edu.cn. 

6S. Zhang is with Department of Data Science, New Jersey Institute of Technology, NJ 07102, USA. E-mail: sz457@njit.edu. 

7D. Ma is with Ocean College, Zhejiang University, Zhoushan, 316021, China; Email: mdf2004@zju.edu.cn. 

† These authors contributed equally to this work. 

B Corresponding author. 

## I. INTRODUCTION

U NMANNED Underwater Vehicles (UUVs) are trans-forming underwater operations, playing critical roles forming underwater operations， playing critical roles in marine research [1], environmental monitoring [2], and resource exploration [3]. However, achieving robust and intelligent autonomy for UUVs, especially in attitude control, remains an open challenge. UUVs operate in complex, highly dynamic, and partially observable environments, where nonlinear hydrodynamics, ocean currents, and wave disturbances introduce significant uncertainty [4]. These factors complicate the design of reliable attitude control systems, which are essential for high-stakes missions such as coral reef navigation [5], pipeline inspection [6], and sample retrieval [7]. 

Traditional and mainstream controllers, such as PID [8], Model Predictive Control (MPC) [9], Sliding Mode Control (SMC) [8], and Fuzzy Logic Control (FLC) [10], provide partial solutions but are often hindered by their reliance on accurate dynamics modeling or their limited adaptability. Their performance often degrades in uncertain conditions due to modeling inaccuracies, hysteresis, and control overshoots, raising risks in unstructured real-world deployments [11], [12]. 

Reinforcement Learning (RL), by contrast, has emerged as a promising data-driven alternative for autonomous agents to learn robust control policies through interaction [13]. In particular, unlike traditional controllers that have inadequate adaptivity or depend on accurate system identification, RL can learn directly from experience, thereby eliminating the need for precise hydrodynamic modeling and enabling end-toend optimization toward task objectives [14]. As a result, RL is especially well-suited for underwater environments, which are nonlinear, partially observable, and difficult to model analytically. Building on this advantage, RL’s adaptability to high-dimensional, nonlinear dynamics has motivated extensive UUV-related research [15]–[17]. Nevertheless, despite these strengths, three major challenges remain: the simulation-toreality (Sim2Real) gap, limited generalizability, and deployment inefficiency. To address these issues, domain randomization can partially mitigate model mismatch by injecting variability into simulation [15]; however, it still cannot fully prevent instability under attitude perturbations or parameter shifts [18]. Moreover, the high computational cost of RL training and the lack of generalizable hydrodynamic/thruster models further constrain its practical application across diverse UUV platforms. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/45a90311043ff7969b6d3f736524b59e8250d8985179a8952da4029583ad2e6c.jpg)



Fig. 1. Illustration of our developed EasyUUV framework. EasyUUV is an LLM-enhanced universal and lightweight Sim2Real RL framework for UUV attitude control, which trains the expert policy via RL in parallelized simulation, while transferring it to a real UUV platform. A multimodal LLM agent further adapts controller parameters using dynamics and sensor feedback for robust performance.


In addition to the challenges described above, the real-world deployment of RL-based controllers often requires extensive manual tuning to handle variations in vehicle dynamics, environmental conditions, and sensor noise, particularly when transitioning across different UUV platforms or operational domains [19]. This lack of adaptability not only hinders scalability but also increases the risk of degraded performance or mission failure in unfamiliar conditions [20]. Fortunately, the introduction of the large language model (LLM) enables online, training-free adaptation of controller parameters [21]. By leveraging historical system trajectories, real-time sensory feedback, and task-specific context encoded in both visual and textual forms, the LLM can dynamically adjust key control parameters without interrupting operations [22]. This capability enhances the robustness and generalizability of the control system, allowing a single RL-trained policy to maintain stable performance across diverse and uncertain underwater environments [23]. 

Based on the above analysis, we developed EasyUUV, a lightweight and universal Sim2Real RL framework enhanced with LLM (as shown in Fig. 1). By combining parallelized RL training with LLM-driven adaptation, it helps bridge the Sim2Real gap for scalable deployment across platforms. The hybrid architecture employs an RL policy for high-level corrections executed by a nonlinear adaptive S-Surface (A-S-Surface) controller, ensuring robust control under 6-DoF coupling. EasyUUV leverages an Isaac Lab [24]-based simulation with hydrodynamic models for efficient parallel training and fast GPU-based convergence. At runtime, a multimodal LLM agent adjusts controller parameters using visual and textual feedback, maintaining performance under unmodeled dynamics, noise, and actuator drift, thus providing a generalizable solution for real-world UUV attitude control. 

Our main contributions are summarized as follows: 

• A universal and lightweight RL-based control framework: EasyUUV supports scalable Sim2Real attitude control through platform-agnostic modeling and a CUDA-accelerated parallelized RL training architecture. To enhance control performance under noise and disturbances, we further develop an A-S-Surface controller that integrates nonlinear control and adaptive compensation for improved robustness. 

• LLM-driven adaptive controller tuning: EasyUUV integrates a multimodal LLM-based module that adaptively adjusts controller parameters at runtime based on historical dynamic responses and real-time sensory feedback, enabling robust adaptation without retraining. 

• Zero-shot transfer and extensive experiments: We develop a low-cost UUV platform integrated with our LLM-enhanced Sim2Real RL framework, enabling zeroshot transfer of expert policies from simulation to reality. Validated through tank experiments and sea trials, EasyUUV showcases outperforming robustness and adaptivity across diverse conditions in attitude control. 

The remainder of this paper is organized as follows: Section II introduces the architecture and core modules of the proposed EasyUUV framework. Section III describes the experimental setup and presents both simulation and real-world results, along with detailed analysis. Finally, Section IV concludes the paper and discusses limitations and future research directions. 

## II. ARCHITECTURE AND MODULES

In this section, we introduce the EasyUUV framework in detail, including both simulation and hardware platforms. 

## A. Architecture Overview

As shown in Fig. 2, the proposed EasyUUV framework consists of three tightly coupled components. The first is a composite control architecture that integrates an RL policy with a nonlinear A-S-Surface controller and an LLM-based adaptation module, enabling both learning-driven decisionmaking and structured stability enforcement. The second component is a parallelized RL training environment that incorporates hydrodynamic effects and thruster models to support large-scale data generation. The third component is a Sim2Real deployment pipeline designed to transfer the trained policy to real-world UUV platforms and support online adaptation during operation. The observation vector provided to the RL policy includes the target attitude, the current estimated attitude, and the depth offset, while the policy outputs deviation commands that are subsequently transformed by the A-S-Surface controller into low-level control inputs and corresponding PWM signals to drive the thrusters. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/9483eb295d263f33eb9e9c14e697f17504b5fc7828013f7d79e4a8804adcc318.jpg)



Fig. 2. Architecture of the EasyUUV framework, which comprise three parts: (a) RL and A-S-Surface-based composite controller module; (b) Parallelized RL training simulation environment developed on Isaac Lab; and (c) Sim2Real deployment module for real-world adaption.


The RL policy is trained in a high-fidelity simulation environment built on NVIDIA Isaac Lab with MuJoCo-based hydrodynamic modeling [25]. During training, domain randomization (DR) is applied to key parameters, including COB-COM offsets [15] and thruster nonlinearities, in order to improve robustness and generalization across varying vehicle dynamics. GPU-accelerated parallel simulation is leveraged to significantly increase sample efficiency and ensure rapid convergence of the policy. Once training is completed, the learned policy is deployed directly to real UUV platforms without additional fine-tuning. During real-world execution, a multimodal LLM agent further enhances adaptability by adjusting controller parameters in real time based on visual observations and textual sensor feedback. This online adaptation mechanism enables robust zero-shot Sim2Real transfer and supports reliable performance across diverse and uncertain underwater operating conditions. 

## B. Simulation Platform and Controller Design

A carefully designed simulation platform is therefore essential for achieving the above capabilities. In the following, we develop a platform that enables efficient RL training and zero-shot policy transfer to real UUVs through simplified, hardware-agnostic modeling integrated with NVIDIA Isaac Lab, while also supporting our LLM-enhanced RL-based control method for robust and adaptive UUV attitude control. 

For hydrodynamic modeling, we adopt MuJoCo-based phenomenological models [25] to simulate rigid-body interactions in fluid. Each object is approximated by an equivalent inertia box computed from its mass m and inertia tensor I, with half-dimensions $r _ { i }$ that can be computed as follows: 

$$
r _ {i} = \sqrt {\frac {3}{2 m} (\mathcal {I} _ {j j} + \mathcal {I} _ {k k} - \mathcal {I} _ {i i})}, \tag {1}
$$

which enables the calculation of total fluid forces $\mathbf { f } _ { \mathrm { i n e r t i a } } =$ $\mathbf { f } _ { D } + \mathbf { f } _ { V }$ and torques $\mathbf { g } _ { \mathrm { i n e r t i a } } = \mathbf { g } _ { D } + \mathbf { g } _ { V }$ , incorporating both drag and viscous effects. Drag forces and torques are modeled as $f _ { D , i } = - 2 \rho r _ { j } r _ { k } | v _ { i } | v _ { i }$ and $\begin{array} { r } { g _ { D , i } = - \frac { 1 } { 2 } \rho r _ { i } ( r _ { j } ^ { 4 } + r _ { k } ^ { 4 } ) | \omega _ { i } | \omega _ { i } } \end{array}$ , while viscous terms are $f _ { V , i } = - 6 \beta \pi r _ { \mathrm { e q } } v _ { i }$ and $g _ { V , i } = - 8 \beta \pi r _ { \mathrm { e q } } ^ { 3 } \omega _ { i }$ , with $r _ { \mathrm { e q } } = ( r _ { x } + r _ { y } + r _ { z } ) / 3$ and $\beta$ denoting the fluid viscosity. 

For thruster dynamics, we implement a realistic actuation pipeline that modulates thrust via PWM signals to electronic speed controllers. Based on empirical data from Blue Robotics T200 thrusters at 16V [26], the thrust output $\tau _ { \Omega }$ (in N) is modeled as a function of normalized input $a ~ \in ~ [ - 1 , 1 ]$ , corresponding to 1100-1900 µs PWM, using a piecewise quadratic fit: 


TABLE I DOMAIN RANDOMIZATION CONFIGURATION DETAILS


<table><tr><td>Parameters</td><td>Distributions</td><td>Values (Low, High)</td></tr><tr><td>COB-COM offset (m)</td><td>Uniform Sphere</td><td>(0.075, 0.15)</td></tr><tr><td>Volume (L)</td><td>Uniform</td><td>(1.5, 3)</td></tr><tr><td>Controller Gain</td><td>Uniform</td><td>(15, 30)% of the relative value</td></tr></table>

$$
\tau_ {\Omega} = \left\{ \begin{array}{l l} 2 9. 5 4 a ^ {2} + 2 6. 1 0 a - 2. 4 4, & a \in (0. 0 8, 1 ], \\ 0, & a \in [ - 0. 0 8, 0. 0 8 ], \\ - 2 1. 7 5 a ^ {2} + 2 1. 7 5 a + 2. 0 7, & a \in [ - 1, - 0. 0 8). \end{array} \right. \tag {2}
$$

To improve policy generalization and real-world adaptability, we apply domain randomization within a high-efficiency, parallelized simulation environment. During training, key parameters such as the COB-COM offset, volume, and controller gains are randomly perturbed to account for structural and dynamic variations.For instance, the COB-COM offset influences the torques induced by gravity and buoyancy, which are critical for attitude control. All simulation models are implemented in Isaac Lab and trained with GPU acceleration. The full set of randomized parameters is listed in Table I. 

Building on this simulation environment, we implement an RL policy using the RSL-RL library [27] with Proximal Policy Optimization (PPO) [28] for training. The UUV observes a 9-dimensional state vector $\vec { o } _ { t } ~ = ~ \{ \vec { q } , \vec { q } \mathrm { { d e s } , \Delta } z \}$ , where $\Delta z$ represents the depth error, while $\vec { q }$ and $\vec { q } _ { \mathrm { d e s } }$ denote the current and desired attitude quaternions, which ensure singularity-free orientation tracking. The policy then outputs a 4-dimensional action vector $\vec { a } _ { t } ~ = ~ \{ \Delta \phi , \Delta \varphi , \Delta \theta , \Delta d \}$ , representing deviations in roll, pitch, yaw, and depth, which are passed to the low-level controller, while a reward function composed of three terms guides the policy toward stable behavior. The terms are listed as follows: 

• $r _ { q } = \exp ( - | \vec { q } \vec { q } _ { \mathrm { d e s } } ^ { * } | )$ encourages orientation alignment, 

• $r _ { p } ~ = ~ \mathrm { e x p } ( - | | \vec { a } | | ^ { b } )$ penalizes excessive control actions (with $b = 1 )$ , 

• $r _ { z } = \exp ( - | | \Delta z | | ^ { 2 } )$ promotes accurate depth tracking. 

These terms are then linearly weighted to guide the policy toward stable and efficient behavior. 

At the low level, we employ an A-S-Surface controller [29] to ensure a fast and robust response under underwater disturbances. Based on the system state $\mathbf { x } ( t ) = [ \delta ( t ) , \dot { \delta } ( t ) ] ^ { \top }$ , the angle error and its derivative are defined as $e ( t ) = \delta _ { \mathrm { d e s } } -$ $\delta ( t )$ and $\dot { e } ( t ) = - \dot { \delta } ( t )$ . The control output is computed as: 

$$
u _ {t} = \frac {2}{1 + \exp (- \zeta_ {1} e (t) - \zeta_ {2} \dot {e} (t))} - 1 + \Delta u (t), \tag {3}
$$

with the adaptive compensation term updated by: 

$$
\Delta u (t + 1) = \Delta u (t) + \alpha e (t) \mathrm{sign} (u _ {t}), \tag {4}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/ed4077f6c4016f15afd393ad5a07436e9295820a8814394627cc0cb1cf7a0ff1.jpg)



Fig. 3. Exploded view of our EasyUUV hardware platform.


where $\alpha$ is a tunable learning rate. This formulation offers both high gain for large deviations and smooth convergence near the setpoint. 

To enable zero-shot transfer in Sim2Real deployment and reduce manual tuning under runtime variations, we incorporate a multimodal LLM that adaptively adjusts controller parameters $( \boldsymbol { \mathrm { e } } . \boldsymbol { \mathrm { g } } . , \zeta _ { 1 } , \zeta _ { 2 } )$ without the need for retraining. Specifically, the LLM processes two types of input: 

• Visual logs, providing a compact representation of control trends and tuning history, conveying complex information concisely while avoiding redundant decisions; 

• Textual data, including sensor readings and user instructions that offer fine-grained, non-visual information. 

These inputs are processed via a lightweight API, with the LLM prompted by context-rich histories and restricted to output adjustment values. Rather than generating new parameters directly, these outputs indicate the direction and scaling factor for modifications. To ensure stability and precision, several fuzzy rules are predefined for scaling factors $( \mathrm { e . g . , 2 \times }$ $/ 0 . 5 \times$ for major changes, $1 . 5 \times / 0 . 6 7 \times$ for finer refinements). This approach enhances numerical stability when handling quantitative inputs [30] and enables timely tuning decisions, thereby improving control robustness in dynamic and various underwater environments. 

## C. Hardware Platform

Our EasyUUV hardware platform (Fig. 3) is a compact, lowcost, and modular testbed designed to support the proposed LLM-enhanced RL framework and to enable Sim2Real zeroshot transfer. The hull integrates 3D-printed ABS components with aluminum structural elements, providing a balance between mechanical durability, corrosion resistance, and rapid prototyping capability. With a total cost of approximately $1000 USD, the platform is significantly more affordable than conventional UUV systems, while its modular architecture allows for flexible integration of sensing, computation, and actuation payloads. The propulsion system consists of eight custom-built thrusters with thrust characteristics comparable to the Blue Robotics T200, arranged in a fully actuated sixdegree-of-freedom configuration to enable independent control of forces and moments along all axes. Vibration isolation is incorporated to mitigate mechanical disturbances and reduce sensor noise. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/5ac71875725fc90e12515f46c0df93cd110c14f0720ebfd1f935e07e24d11da3.jpg)



Fig. 5. Training curves of RL with different controllers in terms of reward and MSE. (Left): Reward curves. (Right): MSE curves.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/1521f55250ee6cf59030cd8ec3019690325df2786c5685bb5ee4ad840106b766.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/e55294fd06f119de18c1d5df9586eb4e6729337f5e2f56a336c6a4b7cc43fa72.jpg)



Fig. 6. Comparison of MSE across two tasks for different controllers, under both RL and non-RL settings. (Left): w/ RL. (Right): w/o RL.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/86da6f2655f1c426ae5101de0667c79f48a2d1c49cf2aa9c7b1bda5005f84af1.jpg)



Fig. 4. Experimental testbed for real-world validation of EasyUUV.


Low-level control is executed by an ESP32-WROOM microcontroller running the proposed A-S-Surface controller at a control frequency of 100 $\mathrm { H z , }$ with controller gains pre-calibrated through simulation-based testing. High-level decision-making is handled by an expert-level RL policy deployed on a surface laptop, with real-time control commands transmitted through a low-latency RS-485 tether. State estimation relies on a 9-DOF MPU9250 IMU, where sensor fusion is implemented using complementary filtering to provide robust and reliable attitude estimation under noisy operating conditions. 

The complete system is housed within a 30 L waterproof enclosure, weighs less than 20 kg, and supports single-person deployment and operation, making it both portable and costeffective for research-grade attitude control experiments. By closely matching the simulated dynamics and leveraging LLMbased online fine-tuning during deployment, EasyUUV provides a practical and reliable platform for evaluating Sim2Real zero-shot transfer in real-world underwater robotic systems. 

## III. EXPERIMENTS

In this section, we describe the experimental setup used for both simulation and real-world testing. As shown in Fig. 4, the EasyUUV testbed consists of UUV hardware connected to a host computer, which performs RL training, policy deployment, and real-time sensor data collection. To mimic realistic underwater dynamics, we also apply two dedicated perturbation generators in a confined indoor tank. 

## A. Simulation Setup

The simulation training was conducted on a computer equipped with a Ryzen 9 7945HX CPU and an RTX 4060 GPU. A total of 460 episodes $( \sim ~ 3 ~ \times ~ 1 0 ^ { 7 }$ steps) were completed in approximately 130 seconds, demonstrating high computational efficiency and rapid policy iteration. 

Toward the end of training, we introduce two evaluation tasks to assess the Mean Square Error (MSE) performance of different control strategies. Task 1 involves tracking a smooth sinusoidal signal, while Task 2 requires following a more complex trajectory constructed by summing multiple sine waves with distinct frequencies: 

$$
s (t) = A \cdot \sum_ {f \in \mathcal {F}} \sin (2 \pi f t), \tag {5}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/bde91fe618c43e36b9b7eeccedf2cabb4ba0692a6dbb13d0f9d448bc2374c69a.jpg)



Fig. 7. Comparison of UUV attitude tracking response curves for different control strategies in simulation experiments. (Left): RL with different controllers. (Right): w/o RL and w/ RL settings.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/5a97af24b112d4ef52be4f7a2340881d922891f22ed9fc2984a8ecfcaff69cb3.jpg)



Fig. 8. Comparison of UUV attitude compound error curves for different control strategies in simulation experiments. (Left): RL with different controllers. (Right): w/o RL and w/ RL settings.


where A is the amplitude (in radians), F is the set of frequencies (in Hz), and t denotes time (in seconds). The specific parameters for each attitude angle are: 

• Yaw: A = 1.35, F = {-0.1, 0.2, 0.4, 0.8, 1.6, -3.2}, 

• Pitch: A = 1.10, F = {-0.1, 0.2, 0.5, -1.0, 2.0, 3.5}, 

• Roll: A = 0.95, F = {0.15, 0.3, 0.5, -0.9, 1.8, -3.0}. 

To further evaluate tracking accuracy, we define the compound error at time t as the sum of absolute differences between actual and desired yaw, pitch, and roll angles extracted from the corresponding quaternions: 

$$
\text { CompoundError } _ {t} = \sum_ {i \in \{\phi , \varphi , \theta \}} | i _ {t} - i _ {\mathrm{des}, t} |. \tag {6}
$$

## B. Simulation Results

We first conduct the simulation RL training, as shown in Fig. 5. The curves compare three controllers-RL with A-S-Surface, S-Surface, and PID-in terms of cumulative reward and MSE. Here, the controller parameters are primarily adopted from [12] to ensure a fair baseline for comparison. In Fig. 5(Left), A-S-Surface converges the fastest and achieves the highest final reward, indicating superior learning efficiency. S-Surface shows slower convergence and lower reward, while PID performs the worst with minimal improvement. In Fig. 5(Right), 

A-S-Surface also maintains the lowest MSE throughout training, followed by S-Surface with moderate error, and PID with consistently the highest MSE. These results highlight the clear advantages of the A-S-Surface controller in adaptive control, particularly in improving learning and tracking performance. 

Building on these observations, Fig. 6 provides a complementary comparison using bar charts of MSE values for the same three controllers across two representative tasks under RL-enabled (w/ RL) and non-RL (w/o RL) settings. In the RL case (Fig. 6(Left)), A-S-Surface consistently achieves the lowest MSE, demonstrating superior tracking accuracy and robustness, while S-Surface shows moderate performance and PID the highest MSE. In the non-RL case (Fig. 6(Right)), all controllers experience a performance drop with a higher MSE; yet A-S-Surface still performs best, indicating that its adaptive structure provides baseline robustness. Overall, these results underscore the combined benefits of RL and adaptive control: RL effectively reduces tracking error, and A-S-Surface remains the most reliable across different conditions. 

Building on this foundation, we next examine the effect of DR on RL training. Specifically, we present MSE results under different DR levels-None (NDR), Small-scale (SDR), and Large-scale (LDR)-to analyze the impact of physical variability on policy generalization. To evaluate out-of-domain performance, we fixed the UUV mass while varying its volume to create two conditions: density at 0.95× and 1.05× the default value (≈ water density), denoted as Pos. buoy and Neg. buoy, respectively. The policies were trained with RL using the A-S-Surface controller. As shown in TABLE II, policies without DR suffer significant MSE degradation under buoyancy shifts, whereas SDR and LDR reduce performance loss, with SDR achieving better generalization and LDR showing less stability. Thus, these findings suggest that exposing policies to broader physical uncertainties during training improves robustness and cross-domain performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/fd87f5073beaccdf7719050eea5244665fa97041a5ac391702f30034703cb999.jpg)



Fig. 9. Comparison of UUV attitude tracking response curves for different attitude angles combination, under both RL and non-RL settings in real-world experiments. (Left): Yaw and Roll. (Right): Yaw and Pitch.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/57094b3bd368f4849d89f0e764c378754c4268543ab272098ad5ceb75aeef8b5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/5b344b20b1e03183f190c9f5d445a3da3bb10e8a1b9bca93f67177477cde0c95.jpg)



Fig. 10. Comparison of UUV attitude compound error curves for different attitude angles combination, under both RL and non-RL setting in realworld experiments. (Left): Yaw and Roll. (Right): Yaw and Pitch.



TABLE II THE MSE RESULTS UNDER VARYING DOMAIN RANDOMIZATION LEVELS IN TASK1 AND TASK2.


<table><tr><td colspan="2">Settings</td><td>NDR</td><td>SDR</td><td>LDR</td></tr><tr><td rowspan="3">Task 1</td><td>In domain</td><td>0.0054</td><td>0.0051</td><td>0.0061</td></tr><tr><td>Pos. buoy</td><td>0.0344</td><td>0.0087</td><td>0.0110</td></tr><tr><td>Neg. buoy</td><td>0.0339</td><td>0.0091</td><td>0.0092</td></tr><tr><td rowspan="3">Task 2</td><td>In domain</td><td>0.0050</td><td>0.0057</td><td>0.0061</td></tr><tr><td>Pos. buoy</td><td>0.0388</td><td>0.0066</td><td>0.0160</td></tr><tr><td>Neg. buoy</td><td>0.0320</td><td>0.0079</td><td>0.0132</td></tr></table>

Having validated robustness against environmental variability, we then turn to attitude tracking performance. To further evaluate attitude tracking in simulation, Fig. 7 compares yaw, pitch, and roll responses under Task 2. In Fig. 7(Left), RL+A-S-Surface achieves the closest tracking with fast convergence and minimal steady-state error, RL+S-Surface shows larger deviations and mild oscillations, while RL+PID responds more slowly with significant errors, especially in pitch and roll. Fig. 7(Right) further shows that A-S-Surface with RL attains higher accuracy and responsiveness than its non-RL counterpart, which exhibits delays and larger errors. Moreover, Fig. 8(Left) illustrates compound error evolution: RL+A-S-Surface maintains the lowest and most stable error, RL+S-Surface shows moderate fluctuations, and RL+PID suffers larger deviations, particularly around 10-12s and near the end. Finally, Fig. 8(Right) confirms that RL reduces both average error (from µ=0.452 to µ=0.103) and variability. Overall, the results highlight the advantage of integrating RL with adaptive control for robust multi-axis attitude regulation. 

## C. Real-World Deployment

Building on the simulation results, we first conduct the tank experiment under disturbance-free conditions to evaluate EasyUUV’s Sim2Real zero-shot transfer capability using the expert-level RL policy directly taken from simulation with the A-S-Surface controller. As shown in Fig. 9, the RLenabled controller tracks desired commands more closely, keeping roll and pitch near the origin with reduced drift and phase lag, while the non-RL case shows larger deviations. In addition, Fig. 10 further compares compound error curves, where the RL-enabled controller achieves lower average error (µ=0.2356 vs. 0.3836, and µ=0.2421 vs. 0.2876) and reduced variability (σ=0.080 vs. 0.150, and σ=0.0896 vs. 0.0970), indicating improved robustness. Taken together, these findings preliminarily confirm that EasyUUV can achieve effective zero-shot transfer from simulation to real-world deployment, significantly enhancing multi-axis tracking performance. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/86b6295024d8d134506b0c668e480a92a764562881aadb02acc4f7a3af436b7b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/5bd093931ead2aaf28512f76b0ad63cc5ec649c8af2b7d5a38e2ecf9f2d259b9.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/ac4347be28438b470b880e1ab846badc784c750f8ecc706fa763ea81c20980c2.jpg)



Fig. 11. Snapshots of EasyUUV operating in a indoor tank at $t = 0 . 0 \mathsf { s } ,$ , 12.5s, and 25.0s in real-world experiments.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/5e33559ff900f60ce092ffd8258e716a3bf3e29721677768abd0a12bd960b712.jpg)



(Left) w/o LLM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/ba84bee2bdcd75b829f64c0b77183f1ce4b2df6e22aec544aa9299156e53c33d.jpg)



(Right) w/ LLM


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/09d4bb2be85a702202a58eae648d5de4a78832be38b7fd12aff9cd8bdcfed93e.jpg)



Fig. 12. Tracking response curves under turbulent perturbations with and without LLM-based online fine-tuning of controller parameters.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/47e20a1712bf1a4eee9a330b47f7d143a62bda2295ce185106bfda65eed02474.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/1803ce0f486b68ecf8ea6590a96d96cdab1c54b8eca9046ab44fe825d3cde43a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/c187c9f17e22c1100d5b4dba78f321e82349a59ffb6009c547be7b1007ebf9d1.jpg)



Fig. 14. Snapshots of EasyUUV operating in real ocean conditions at $t = 0 . 0 \mathsf { s } ,$ 12.5s, and 25.0s in sea trials.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/79868e457ff703abf8ae81d78f5a02f771dea5fe5c5e3ba8aaba2cab634591af.jpg)



Fig. 13. Tracking response curves along the roll, pitch, and yaw axes in tank experiments under turbulent and transient perturbation.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-06-07/b71c8e1a-8270-415b-b44d-c26cb166a5a3/bb14312062c93e620454df7af3ed27a2a1ec9e74f3084c8d04c0ff6607fd5299.jpg)



Fig. 15. Tracking response curves along the roll, pitch, and yaw axes in sea trials under wave-induced turbulence.


After the initial disturbance-free tests, we further activated the perturbation generators to validate the effectiveness of the LLM’s online fine-tuning capacity. As shown in Fig. 11, EasyUUV rapidly suppresses disturbances and restores the vehicle to the desired trajectory, demonstrating strong robustness in real-world conditions. Fig. 12 further evaluates yaw tracking under turbulence, where LLM-based online parameter tuning progressively reduces the mean squared error from 0.0812 rad2 to 0.0179 rad2 after two adjustments, significantly enhancing tracking accuracy and stability. These results confirm that EasyUUV not only withstands perturbations but also leverages LLM-based adjustments for adaptive, highprecision control in dynamic underwater environments. 

To more rigorously assess robustness, two strong transient perturbations are manually introduced at 10.3s and 19.4s. As illustrated in Fig. 13, under turbulent disturbances, the EasyUUV still tracks the desired trajectory closely on all three axes with steady-state errors near zero. Although roll and pitch briefly deviate when the manual perturbations occur, the framework quickly suppresses the errors and restores the trajectory, while yaw tracking remains accurate throughout. These results verify the robustness and stability of the proposed framework, enabling EasyUUV to maintain highprecision control under turbulent and sudden disturbances while demonstrating strong Sim2Real transfer capability. 



[11] C.-W. Chen, N.-M. Yan, J.-X. Leng, and Y. Chen, “Numerical analysis of second-order wave forces acting on an autonomous underwater helicopter using panel method,” in OCEANS 2017 - Anchorage, pp. 1–6, 2017. 



Based on the above tests, we finally extend the evaluation to sea trials. Figs. 14 and 15 present the results, complementing the earlier tank experiments and highlighting the framework’s zero-shot domain transfer capability. The tracking curves indicate that EasyUUV closely follows the desired commands in roll, pitch, and yaw under wave-induced turbulence, with steady-state errors near zero. In addition, snapshots at t=0.0s, 12.5s, and 25.0s clearly illustrate operation in real ocean conditions with waves and strong flows. Together with the tank experiments, these results confirm that the framework transfers directly from controlled environments to open-sea settings without retraining, achieving robust disturbance rejection and stable high-precision control. 



[12] G. Xie, J. Xu, Y. Ding, Z. Zhang, S. Zhang, and Y. Li, “Never too prim to swim: An llm-enhanced rl-based adaptive s-surface controller for auvs under extreme sea conditions,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), DOI 10.1109/IROS60139.2025.11247231, pp. 8990–8997, 2025. 



## IV. CONCLUSIONS



[13] W. Wei, J. Wang, J. Du, Z. Fang, Y. Ren, and C. L. P. Chen, “Differential game-based deep reinforcement learning in underwater target hunting task,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, DOI 10.1109/TNNLS.2023.3325580, no. 1, pp. 462–474, 2025. 



In this paper, we introduce EasyUUV, an LLM-enhanced universal and lightweight Sim2Real RL framework for robust UUV attitude control. The framework integrates domainrandomized RL training with a hybrid control architecture that incorporates an A-S-Surface controller, while a multimodal LLM agent provides runtime parameter fine-tuning without additional retraining. Built on a cost-effective 6- DoF UUV platform, EasyUUV enables efficient simulationbased policy learning and achieves zero-shot transfer to realworld deployment. Extensive simulation and field experiments demonstrate that EasyUUV offers stable, generalizable control, with superior robustness and consistent Sim2Real performance under diverse underwater conditions. 



[14] I. Masmitja, M. Martin, T. O’Reilly, B. Kieft, N. Palomeras, J. Navarro, and K. Katija, “Dynamic robotic tracking of underwater targets using reinforcement learning,” Science Robotics, vol. 8, no. 80, p. eade7811, 2023. 



Limitations: While EasyUUV demonstrates encouraging Sim2Real robustness for underwater vehicle attitude control, several limitations remain. First, the transfer performance is still inherently constrained by the fidelity of the simulation environment and the scope of domain randomization. Although multiple sources of uncertainty are introduced, factors such as long-term actuator degradation, slowly varying sensor bias, and communication-induced delays are not yet fully captured. 



[15] L. Cai, K. Chang, and Y. Girdhar, “Learning to swim: Reinforcement learning for 6-dof control of thruster-driven autonomous underwater vehicles,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), DOI 10.1109/ICRA55743.2025.11128688, pp. 11 286– 11 293, 2025. 



Second, the LLM-driven online tuning mechanism relies on structured prompts and heuristic, ratio-based parameter adjustments. While this design improves flexibility and deployment convenience, it also introduces a degree of non-determinism and limits the ability to provide strict formal guarantees on closed-loop stability under all operating conditions. Finally, the experimental evaluation is conducted on a specific lowcost UUV platform and focuses primarily on attitude tracking tasks, leaving broader generalization across platforms, longerduration missions, and additional performance aspects such as energy efficiency and fault tolerance insufficiently explored. 



[16] D. Meger, J. C. G. Higuera, A. Xu, P. Giguere, and G. Dudek, “Learning ` legged swimming gaits from experience,” in 2015 IEEE International Conference on Robotics and Automation (ICRA), pp. 2332–2338, 2015. 



Future work: Building on these limitations, future work will focus on extending EasyUUV beyond low-level attitude regulation toward higher-level autonomy. In particular, we plan to integrate visual and language-based models with underwater image enhancement techniques, enabling more reliable perception and task interpretation in turbid and low-visibility environments. At the system level, we will investigate more self-contained onboard deployment by optimizing inference efficiency and introducing a safety supervision layer that constrains or overrides LLM-guided parameter updates when abnormal dynamics are detected. In addition, we aim to conduct broader evaluations across multiple UUV platforms and longer-term field trials, while explicitly incorporating delays, faults, and slow environmental drift into the uncertainty set. These efforts are expected to further improve generalization and provide a more comprehensive understanding of robustness in realistic underwater operations. 



[17] B. Hadi, A. Khosravi, and P. Sarhadi, “Deep reinforcement learning for adaptive path planning and control of an autonomous underwater vehicle,” Applied Ocean Research, vol. 129, p. 103326, 2022. 



## REFERENCES



[18] D. Xue, Z. Gengshi, X. Jian, and C. Tao, “Position and attitude control of uuv in the process of operation tasks,” in 2018 37th Chinese Control Conference (CCC), pp. 2893–2898, 2018. 





[19] V. Sufan and G. Troni, “Swim4real: Deep reinforcement learning- ´ based energy-efficient and agile 6-dof control for underwater vehicles,” IEEE Robotics and Automation Letters, vol. 10, DOI 10.1109/LRA.2025.3575650, no. 7, pp. 7326–7333, 2025. 





[1] L. Hawkes, O. Exeter, S. Henderson, C. Kerry, A. Kukulya, J. Rudd, S. Whelan, N. Yoder, and M. Witt, “Autonomous underwater videography and tracking of basking sharks,” Animal Biotelemetry, vol. 8, Aug. 2020. 





[20] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from sim-





[2] J. Rutledge, W. Yuan, J. Wu, S. Freed, A. Lewis, Z. Wood, T. Gambin, and C. Clark, “Intelligent shipwreck search using autonomous underwater vehicles,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6175–6182, 2018. 





[3] F. L. Pena, F. Orjales, and A. Deibe, “Development of a collaborative ˜ host-guest unmanned underwater vehicle docking system for inspection and maintenance of offshore structures,” in OCEANS 2023 - MTS/IEEE U.S. Gulf Coast, pp. 1–5, 2023. 





ulation to the real world,” in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 23–30, 2017. 





[4] X. Lin, N. J. Sanket, N. Karapetyan, and Y. Aloimonos, “Oysternet: Enhanced oyster detection using simulation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), DOI 10.1109/ICRA48891.2023.10160830, pp. 5170–5176, 2023. 





[21] R. Zahedifar, M. Soleymani Baghshah, and A. Taheri, “Llmcontroller: Dynamic robot control adaptation using large language models,” Robotics and Autonomous Systems, vol. 186, DOI https://doi.org/10.1016/j.robot.2024.104913, p. 104913, 2025. 





[5] X. Lin, N. Karapetyan, K. Joshi, T. Liu, N. Chopra, M. Yu, P. Tokekar, and Y. Aloimonos, “Uivnav: Underwater information-driven visionbased navigation via imitation learning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 5250–5256, 2024. 





[22] J. Xu, Z. Zheng, and Z. Wang, “Lac: Using llm-based agents as the controller to realize embodied robot,” in 2024 IEEE International Conference on Robotics and Biomimetics (ROBIO), pp. 1894–1899, 2024. 





[6] J. WEI, “Auv optical vision submarine pipeline inspection,” in 2024 7th International Symposium on Autonomous Systems (ISAS), pp. 1–5, 2024. 





[23] Q. Guo, X. Liu, J. Hui, Z. Liu, and P. Huang, “Utilizing large language models for robot skill reward shaping in reinforcement learning,” in Intelligent Robotics and Applications, pp. 3–17, Singapore, 2025. 





[7] I. Salman, N. Karapetyan, A. Venkatachari, A. Q. Li, A. Bourbonnais, and I. Rekleitis, “Multi-modal lake sampling for detecting harmful algal blooms,” in OCEANS 2022, Hampton Roads, pp. 1–9, 2022. 





[24] M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, “Orbit: A unified simulation framework for interactive robot learning environments,” IEEE Robotics and Automation Letters, vol. 8, DOI 10.1109/LRA.2023.3270034, no. 6, pp. 3740–3747, 2023. 





[8] A. Mitchell, E. McGookin, and D. Murray-Smith, “Comparison of control methods for autonomous underwater vehicles,” IFAC Proceedings Volumes, vol. 36, no. 4, pp. 37–42, 2003, iFAC Workshop on Guidance and Control of Underwater Vehicles 2003, Newport, South Wales, UK, 9-11 April 2003. 





[25] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for modelbased control,” in 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 5026–5033. IEEE, 2012. 





[9] Z. Yan, P. Gong, W. Zhang, and W. Wu, “Model predictive control of autonomous underwater vehicles for trajectory tracking with external disturbances,” Ocean Engineering, vol. 217, p. 107884, 2020. 





[26] BlueRobotics, “Bluerobotics t200 performace charts,” 2019. [Online]. Available: https://cad.bluerobotics.com/ T200-Public-Performance-Data-10-20V-September-2019.xlsx 





[10] B. Hu, H. Tian, J. Qian, G. Xie, L. Mo, and S. Zhang, “A fuzzy-pid method to improve the depth control of auv,” in 2013 IEEE International Conference on Mechatronics and Automation, pp. 1528–1533, 2013. 





[27] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes using massively parallel deep reinforcement learning,” in Conference on Robot Learning, pp. 91–100. PMLR, 2022. 





[28] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017. 





[29] B. Li, Y. Xu, C. Liu, and W. Xu, “Simulation and preliminary experimental results on s-surface control of an autonomous underwater vehicle based on moos-ivp,” in 2014 Oceans - St. John’s, pp. 1–6, 2014. 





[30] Y. Yan, Y. Lu, R. Xu, and Z. Lan, “Do phd-level llms truly grasp elementary addition? probing rule learning vs. memorization in large language models,” arXiv preprint arXiv:2504.05262, 2025. 

