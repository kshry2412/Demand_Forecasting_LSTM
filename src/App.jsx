import React, { useState } from 'react';
import Login from './components/Login';
import Register from './components/Register';
import './App.css';

function App() {
  // 'isLogin' is true by default. When it's false, we show Register.
  const [isLogin, setIsLogin] = useState(true);

  const togglePage = (e) => {
    e.preventDefault();
    setIsLogin(!isLogin);
  };

  return (
    <div className="App">
      {isLogin ? (
        <Login onSwitch={togglePage} />
      ) : (
        <Register onSwitch={togglePage} />
      )}
    </div>
  );
}

export default App;