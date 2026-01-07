import React from 'react';

const Register = ({ onSwitch }) => {
    return (
        <div className="container">
            <div className="left-side">
                <div className="branding">
                    <div className="logo-icon">✦</div>
                    <h1>MedLink AI</h1>
                    <p>Register your credentials to calibrate the LSTM forecasting engine.</p>
                </div>
            </div>
            <div className="right-side">
                <div className="glass-card">
                    <form onSubmit={(e) => e.preventDefault()}>
                        <h2>Professional Profile</h2>
                        <div className="input-row">
                            <div className="input-group">
                                <label>Full Name</label>
                                <input type="text" placeholder="John Doe" required />
                            </div>
                            <div className="input-group">
                                <label>Employee ID</label>
                                <input type="text" placeholder="EMP-2026" required />
                            </div>
                        </div>
                        <div className="input-group">
                            <label>Department</label>
                            <select className="custom-select">
                                <option>Pharmacy</option>
                                <option>Surgical Units</option>
                                <option>Procurement</option>
                            </select>
                        </div>
                        <div className="input-group">
                            <label>Password</label>
                            <input type="password" required />
                        </div>
                        <button type="submit" className="login-btn">Finalize Registration</button>
                        <p className="signup-text">
                            Already have an account? <a href="#" onClick={onSwitch}>Sign In</a>
                        </p>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Register;