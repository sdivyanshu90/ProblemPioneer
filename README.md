# ML4SCI Task Solutions Repository

Welcome! This repository contains solutions to three tasks provided by ML4SCI (Machine Learning for Science). Let's take a look at each task:

## Task Details

1. **Electron/Photon Classification:**
   - Classifying electrons and photons in particle physics experiments using machine learning algorithms. We'll develop models to accurately distinguish between these two particle types based on their characteristics.

   **Dataset Description:**
   - 32x32 matrices with two channels:
     - Hit energy
     - Time
   - Two types of particles:
     - Electrons
     - Photons
   - These matrices represent the particles hitting the detector.

2. **Deep Learning-based Quark-Gluon Classification:**
   - Classifying quarks and gluons using deep learning techniques. Quarks and gluons are fundamental particles in the Standard Model of particle physics, and accurate classification is crucial for various physics analyses.

   **Dataset Description:**
   - 125x125 matrices in three-channel images.
   - Two classes of particles:
     - Quarks
     - Gluons
   - The images depict particles impinging on a calorimeter.

3. **CMS Project Task 3g:**
   - This task focuses on particle momentum estimation in the CMS (Compact Muon Solenoid) project. We'll explore graph neural networks (GNN) for momentum regression in the CMS Trigger System.

   **Discussion:**
   - Both GCN (Graph Convolutional Network) and GAT (Graph Attention Network) models, which work with graphs, show good results when compared to traditional methods like CNNs. However, to make these models work well, it's important to create good graphs. This means we need to explore better ways to build graphs to improve the models in the future.

   - When it comes to how fast they work, GCN usually works faster than GAT. This is because GCN has a simpler design and works in a more straightforward way. It takes less time to train and doesn't need as much computing power. But even though GCN is faster, it can still perform as well as GAT, which is more complex. This shows that both methods are effective in capturing relationships in graphs.

   **Dataset Description:**
   - Identical to that of Common Task 2.
   - 125x125 matrices in three-channel images, depicting particles impinging on a calorimeter.

Thank you for visiting the ML4SCI Task Solutions Repository!
