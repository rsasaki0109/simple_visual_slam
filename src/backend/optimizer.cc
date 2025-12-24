#include "backend/optimizer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/manifold.h>
#include <ceres/product_manifold.h>

namespace svslam {

// Define Cost Functions here

// Reprojection Error Cost Function
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, double fx, double fy, double cx, double cy)
        : observed_x(observed_x), observed_y(observed_y), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const camera_pose, // [tx, ty, tz, qx, qy, qz, qw]
                    const T* const point,       // [x, y, z]
                    T* residuals) const {
        
        // Transform point from world to camera: P_c = R * P_w + t
        // camera_pose[0-2] is t
        // camera_pose[3-6] is q (x, y, z, w)
        
        T p[3];
        ceres::QuaternionRotatePoint(camera_pose + 3, point, p);
        p[0] += camera_pose[0];
        p[1] += camera_pose[1];
        p[2] += camera_pose[2];
        
        // Projection
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        
        T predicted_x = T(fx) * xp + T(cx);
        T predicted_y = T(fy) * yp + T(cy);
        
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        
        return true;
    }
    
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y,
                                       const double fx,
                                       const double fy,
                                       const double cx,
                                       const double cy) {
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 7, 3>(
            new ReprojectionError(observed_x, observed_y, fx, fy, cx, cy)));
    }
    
    double observed_x;
    double observed_y;
    double fx, fy, cx, cy;
};

void Optimizer::bundleAdjustment(const std::vector<Keyframe::Ptr>& keyframes, 
                                 const std::vector<Landmark::Ptr>& landmarks,
                                 int iterations) {
    ceres::Problem problem;
    
    // Parameter blocks
    // 1. Keyframe Poses
    // We use double[7] for T_cw (translation + quaternion)
    // Map: KF ID -> double*
    std::map<unsigned long, double*> pose_params;
    
    for (auto& kf : keyframes) {
        double* param = new double[7];
        Eigen::Vector3d t = kf->T_cw_.translation();
        Eigen::Quaterniond q = kf->T_cw_.unit_quaternion();
        
        param[0] = t.x();
        param[1] = t.y();
        param[2] = t.z();
        param[3] = q.x();
        param[4] = q.y();
        param[5] = q.z();
        param[6] = q.w();
        
        pose_params[kf->id_] = param;
        
        // Construct Manifold for SE3 (Euclidean<3> x EigenQuaternion)
        ceres::Manifold* manifold = new ceres::ProductManifold<ceres::EuclideanManifold<3>, ceres::EigenQuaternionManifold>();
        
        problem.AddParameterBlock(param, 7, manifold);
        
        if (kf->id_ == 0) {
            problem.SetParameterBlockConstant(param); // Fix first frame
        }
    }
    
    // 2. Landmarks
    std::map<unsigned long, double*> point_params;
    
    for (auto& lm : landmarks) {
        if (lm->isBad()) continue;
        
        double* param = new double[3];
        Vec3 pos = lm->getPos();
        param[0] = pos.x();
        param[1] = pos.y();
        param[2] = pos.z();
        
        point_params[lm->id_] = param;
        
        problem.AddParameterBlock(param, 3);
        
        // Add Residuals
        // Iterate observations
        for (auto& obs : lm->observations_) {
            auto kf = obs.first.lock();
            if (!kf) continue;
            
            // Only if kf is in our optimization set (Local BA)
            if (pose_params.find(kf->id_) == pose_params.end()) {
                // If KF is fixed (neighbor), we still add residual but keep KF constant
                // But for simplicity in this function, we assume all involved KFs are passed in 'keyframes' argument
                // Or we can handle 'fixed' keyframes if we fetch them from observations.
                // Let's strictly use keyframes passed in.
                continue; 
            }
            
            int kp_idx = obs.second;
            const auto& kp = kf->keypoints_[kp_idx];
            
            auto camera = kf->camera_;
            
            ceres::CostFunction* cost_function = ReprojectionError::Create(
                kp.pt.x, kp.pt.y,
                camera->fx_, camera->fy_, camera->cx_, camera->cy_);
                
            problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), pose_params[kf->id_], point_params[lm->id_]);
        }
    }
    
    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = iterations;
    options.minimizer_progress_to_stdout = true;
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.BriefReport() << std::endl;
    
    // Update State
    for (auto& kf : keyframes) {
        double* param = pose_params[kf->id_];
        Eigen::Vector3d t(param[0], param[1], param[2]);
        Eigen::Quaterniond q(param[6], param[3], param[4], param[5]); // w, x, y, z
        
        kf->T_cw_ = SE3(q, t);
        
        delete[] param;
    }
    
    for (auto& lm : landmarks) {
        if (point_params.count(lm->id_)) {
            double* param = point_params[lm->id_];
            Vec3 pos(param[0], param[1], param[2]);
            lm->setPos(pos);
            delete[] param;
        }
    }
}

int Optimizer::poseOptimization(Frame::Ptr frame) {
    // Stub
    return 0;
}

}
