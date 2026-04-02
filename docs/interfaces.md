# Current Minimal Interface

## Reference Keyframe Policy

```cpp
struct ReferenceKeyframePolicyInput {
    int tracked_features;
    int detected_keypoints;
    int candidate_landmarks;
    int frames_since_reference;
    int lost_frames;
    bool has_depth;
    bool has_accel;
};

struct ReferenceKeyframeDecision {
    ReferenceKeyframeAction action;
    double confidence;
    std::string reason;
};
```

## Why These Fields Survived

- `tracked_features`: every policy needs current tracking density.
- `detected_keypoints`: separates sparse-image failure from healthy mono frames.
- `candidate_landmarks`: lets policies reason about how much map support the candidate keyframe brings.
- `frames_since_reference`: makes keyframe aging explicit instead of hidden inside each implementation.
- `lost_frames`: keeps recovery pressure observable to the policy instead of coupling it to `Tracking` internals.
- `has_depth`: the primary modality flag that every implementation already relied on.
- `has_accel`: survived once the room follow-up showed `depth` and `depth_accel` windows do not behave the same under replay.

## Why Other Fields Were Removed

- Raw `Frame`, `Keyframe`, and `Map` pointers were excluded because they destroy comparability and make policy evaluation depend on global SLAM state.
- Optimizer, loop-closing, and descriptor state were excluded because none of the current variants needed them to make a comparable decision.
