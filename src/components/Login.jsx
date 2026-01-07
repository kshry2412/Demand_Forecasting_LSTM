import React, { useState } from 'react';

const Login = ({ onSwitch }) => {
    const [isLoading, setIsLoading] = useState(false);

    const handleSubmit = (e) => {
        e.preventDefault();
        setIsLoading(true);
        setTimeout(() => {
            alert("Authenticated! Opening AI Dashboard...");
            setIsLoading(false);
        }, 1500);
    };

    return (
        <div className="container">
            <div className="left-side">
                <div className="branding">
                    <h1>MedLink AI</h1>
                    <p>Advanced Demand Forecasting. Utilizing LSTM models to optimize medical supply chains.</p>
                    <div className="stats-badge">Neural Network Status: Active</div>
                </div>
            </div>
            <div className="right-side">
                <div className="glass-card">
                    <form onSubmit={handleSubmit}>
                        <h2>Welcome Back</h2>
                        <p className="subtitle">Enter your credentials.</p>
                        <div className="input-group">
                            <label>Email</label>
                            <input type="email" placeholder="name@medical-org.com" required />
                        </div>
                        <div className="input-group">
                            <label>Password</label>
                            <input type="password" placeholder="••••••••" required />
                        </div>
                        <button type="submit" className="login-btn" disabled={isLoading}>
                            {isLoading ? "Authenticating..." : "Sign In"}
                        </button>
                        <p className="signup-text">
                            New here? <a href="#" onClick={onSwitch}>Create Account</a>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Login;