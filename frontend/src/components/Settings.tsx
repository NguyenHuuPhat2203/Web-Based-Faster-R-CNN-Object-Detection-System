import React, { useState } from "react";
import {
  User,
  Mail,
  Lock,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  ShieldAlert,
  LogOut,
  Info,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { messageOf, updateProfile, changePassword, type ProfileUpdate } from "../lib/api";

export function Settings() {
  const { logout, user, setUser } = useAuth();

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
      const data: ProfileUpdate = {};
      if (username.trim()) data.username = username.trim();
      if (email.trim()) data.email = email.trim();
      if (Object.keys(data).length === 0) {
        setProfileMsg({ type: "error", text: "No changes to save." });
        setProfileSubmitting(false);
        return;
      }
      setUser(await updateProfile(data));
      setProfileMsg({ type: "success", text: "Profile updated successfully!" });
      setUsername("");
      setEmail("");
    } catch (err) {
      setProfileMsg({ type: "error", text: messageOf(err, "Update failed") });
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
    } catch (err) {
      setPasswordMsg({ type: "error", text: messageOf(err, "Password change failed") });
    } finally {
      setPasswordSubmitting(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Settings</h1>

      {/* Profile */}
      <div className="settings-card">
        <h2 className="settings-section-title"><User size={18} /> Profile</h2>
        {profileMsg && (
          <div className={`auth-${profileMsg.type === "error" ? "error" : "success"}`}>
            {profileMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{profileMsg.text}</span>
          </div>
        )}
        <form onSubmit={handleProfileSubmit}>
          <div className="input-group">
            <User size={18} />
            <input
              type="text"
              placeholder={`Username${user ? ` (current: ${user.username})` : ""}, leave blank to keep`}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              minLength={2}
            />
          </div>
          <div className="input-group">
            <Mail size={18} />
            <input
              type="email"
              placeholder={`Email${user ? ` (current: ${user.email})` : ""}, leave blank to keep`}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="settings-actions">
            <button className="btn btn-primary" type="submit" disabled={profileSubmitting}>
              {profileSubmitting ? "Saving…" : "Save Changes"}
            </button>
          </div>
        </form>
      </div>

      {/* Password */}
      <div className="settings-card">
        <h2 className="settings-section-title"><Lock size={18} /> Change Password</h2>
        {passwordMsg && (
          <div className={`auth-${passwordMsg.type === "error" ? "error" : "success"}`}>
            {passwordMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{passwordMsg.text}</span>
          </div>
        )}
        <form onSubmit={handlePasswordSubmit}>
          <div className="input-group">
            <Lock size={18} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="Current password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <Lock size={18} />
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
            <Lock size={18} />
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
              {passwordSubmitting ? "Updating…" : "Update Password"}
            </button>
            <button type="button" className="btn btn-ghost" style={{ width: "auto" }}
              onClick={() => setShowPasswords(!showPasswords)}>
              {showPasswords ? <EyeOff size={18} /> : <Eye size={18} />}
              {showPasswords ? "Hide" : "Show"} Passwords
            </button>
          </div>
        </form>
      </div>

      {/* Account / Danger Zone */}
      <div className="settings-card danger-zone">
        <h2 className="settings-section-title"><ShieldAlert size={18} /> Account</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <Info size={16} style={{ color: "var(--text-secondary)", flexShrink: 0 }} />
            <div>
              <p style={{ fontSize: "0.82rem", color: "var(--text)" }}>
                {user ? `Signed in as ${user.email}` : "NeuroScan AI"}
              </p>
            </div>
          </div>
          <div className="settings-actions">
            <button className="btn btn-danger" onClick={logout} style={{ width: "auto" }}>
              <LogOut size={16} /> Sign Out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
