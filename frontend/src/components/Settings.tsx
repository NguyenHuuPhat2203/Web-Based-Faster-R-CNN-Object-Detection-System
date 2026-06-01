import React, { useState } from "react";
import {
  User,
  Mail,
  Lock,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
} from "lucide-react";
import { updateProfile, changePassword } from "../lib/api";

export function Settings() {
  // Profile state
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [profileMsg, setProfileMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [profileSubmitting, setProfileSubmitting] = useState(false);

  // Password state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPasswords, setShowPasswords] = useState(false);
  const [passwordMsg, setPasswordMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [passwordSubmitting, setPasswordSubmitting] = useState(false);

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileMsg(null);
    setProfileSubmitting(true);
    try {
      const data: Record<string, string> = {};
      if (username.trim()) data.username = username.trim();
      if (email.trim()) data.email = email.trim();
      if (Object.keys(data).length === 0) {
        setProfileMsg({ type: "error", text: "No changes to save." });
        setProfileSubmitting(false);
        return;
      }
      await updateProfile(data);
      setProfileMsg({ type: "success", text: "Profile updated successfully!" });
      setUsername("");
      setEmail("");
    } catch (err: any) {
      setProfileMsg({ type: "error", text: err.message || "Update failed" });
    } finally {
      setProfileSubmitting(false);
    }
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordMsg(null);

    if (newPassword !== confirmPassword) {
      setPasswordMsg({ type: "error", text: "Passwords do not match" });
      return;
    }
    if (newPassword.length < 6) {
      setPasswordMsg({ type: "error", text: "Password must be at least 6 characters" });
      return;
    }

    setPasswordSubmitting(true);
    try {
      await changePassword(currentPassword, newPassword);
      setPasswordMsg({ type: "success", text: "Password changed successfully!" });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err: any) {
      setPasswordMsg({ type: "error", text: err.message || "Password change failed" });
    } finally {
      setPasswordSubmitting(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Settings</h1>

      {/* Profile */}
      <div className="settings-card">
        <h2 className="settings-section-title"><User size={20} /> Profile</h2>
        {profileMsg && (
          <div className={`auth-${profileMsg.type === "error" ? "error" : "success"}`}>
            {profileMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{profileMsg.text}</span>
          </div>
        )}
        <form onSubmit={handleProfileSubmit}>
          <div className="input-group">
            <User size={20} />
            <input
              type="text"
              placeholder="New username (leave blank to keep current)"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              minLength={2}
            />
          </div>
          <div className="input-group">
            <Mail size={20} />
            <input
              type="email"
              placeholder="New email (leave blank to keep current)"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <button className="btn btn-primary" type="submit" disabled={profileSubmitting}>
            {profileSubmitting ? "Saving&hellip;" : "Save Changes"}
          </button>
        </form>
      </div>

      {/* Password */}
      <div className="settings-card">
        <h2 className="settings-section-title"><Lock size={20} /> Change Password</h2>
        {passwordMsg && (
          <div className={`auth-${passwordMsg.type === "error" ? "error" : "success"}`}>
            {passwordMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{passwordMsg.text}</span>
          </div>
        )}
        <form onSubmit={handlePasswordSubmit}>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="Current password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="New password (min 6 chars)"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="Confirm new password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>
          <div className="settings-actions">
            <button className="btn btn-primary" type="submit" disabled={passwordSubmitting}>
              {passwordSubmitting ? "Updating&hellip;" : "Update Password"}
            </button>
            <button type="button" className="btn btn-ghost"
              onClick={() => setShowPasswords(!showPasswords)}>
              {showPasswords ? <EyeOff size={18} /> : <Eye size={18} />}
              {showPasswords ? "Hide" : "Show"} Passwords
            </button>
          </div>
        </form>
      </div>

      {/* Account Info */}
      <div className="settings-card">
        <h2 className="settings-section-title"><User size={20} /> Account Info</h2>
        <p className="text-muted">View-only account details.</p>
      </div>
    </div>
  );
}
