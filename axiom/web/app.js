const state = {
  setup: null,
  sources: [],
  activities: [],
  workspace: null,
  restoringChat: false,
  contextItems: [
    { id: "ctx-subject", type: "subject", label: "Subject", value: "international development" },
    { id: "ctx-root", type: "root", label: "Root", value: "samples" },
  ],
  widgetSerial: 0,
  questionSerial: 0,
  attachmentSerial: 0,
  pendingAttachments: [],
  lastSubject: "international development",
};

const WORKSPACE_STORAGE_KEY = "axiom.workspace.v1";

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let data = {};
  try {
    data = await response.json();
  } catch (err) {
    data = { error: response.statusText || String(err) };
  }
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function jsonText(value) {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function commandParts(input) {
  return input.match(/(?:[^\s"]+|"[^"]*")+/g)?.map((part) => part.replace(/^"|"$/g, "")) || [];
}

function uniqueId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function folderNameFromPath(path) {
  const clean = String(path || "samples").replace(/[\\/]+$/g, "");
  return clean.split(/[\\/]/).filter(Boolean).pop() || clean || "Workspace";
}

function chatTitleFromText(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  if (!clean) return "New chat";
  return clean.length > 42 ? `${clean.slice(0, 39)}...` : clean;
}

function defaultWorkspace() {
  const folderId = uniqueId("folder");
  const chatId = uniqueId("chat");
  const createdAt = nowIso();
  return {
    version: 1,
    activeFolderId: folderId,
    activeChatId: chatId,
    folders: [
      {
        id: folderId,
        name: "samples",
        path: "samples",
        createdAt,
        updatedAt: createdAt,
        chats: [
          {
            id: chatId,
            title: "New chat",
            createdAt,
            updatedAt: createdAt,
            messages: [],
          },
        ],
      },
    ],
  };
}

function normalizeWorkspace(candidate) {
  const fallback = defaultWorkspace();
  if (!candidate || typeof candidate !== "object") return fallback;
  const folders = Array.isArray(candidate.folders) ? candidate.folders : [];
  const cleanedFolders = folders
    .filter((folder) => folder && typeof folder === "object")
    .map((folder) => {
      const id = String(folder.id || uniqueId("folder"));
      const path = String(folder.path || folder.name || "samples");
      const chats = Array.isArray(folder.chats)
        ? folder.chats
            .filter((chat) => chat && typeof chat === "object")
            .map((chat) => ({
              id: String(chat.id || uniqueId("chat")),
              title: String(chat.title || "New chat"),
              createdAt: String(chat.createdAt || nowIso()),
              updatedAt: String(chat.updatedAt || chat.createdAt || nowIso()),
              messages: Array.isArray(chat.messages) ? chat.messages.slice(-200) : [],
            }))
        : [];
      return {
        id,
        name: String(folder.name || folderNameFromPath(path)),
        path,
        fileCount: Number(folder.fileCount || 0),
        createdAt: String(folder.createdAt || nowIso()),
        updatedAt: String(folder.updatedAt || nowIso()),
        chats,
      };
    });
  if (!cleanedFolders.length) return fallback;
  const activeFolder = cleanedFolders.find((folder) => folder.id === candidate.activeFolderId) || cleanedFolders[0];
  const activeChat = activeFolder.chats.find((chat) => chat.id === candidate.activeChatId) || activeFolder.chats[0] || null;
  return {
    version: 1,
    activeFolderId: activeFolder.id,
    activeChatId: activeChat?.id || "",
    folders: cleanedFolders,
  };
}

function loadWorkspace() {
  try {
    state.workspace = normalizeWorkspace(JSON.parse(localStorage.getItem(WORKSPACE_STORAGE_KEY) || "null"));
  } catch (err) {
    state.workspace = defaultWorkspace();
  }
  saveWorkspace();
}

function saveWorkspace() {
  if (!state.workspace) return;
  try {
    localStorage.setItem(WORKSPACE_STORAGE_KEY, JSON.stringify(state.workspace));
  } catch (err) {
    addActivity("Workspace not saved", "Browser storage is full or unavailable");
  }
}

function activeFolder() {
  if (!state.workspace) return null;
  return state.workspace.folders.find((folder) => folder.id === state.workspace.activeFolderId) || null;
}

function activeChat() {
  const folder = activeFolder();
  if (!folder) return null;
  return folder.chats.find((chat) => chat.id === state.workspace.activeChatId) || null;
}

function ensureWorkspaceFolder(path, options = {}) {
  if (!state.workspace) return null;
  const cleanPath = String(path || "samples").trim() || "samples";
  let folder = state.workspace.folders.find((item) => item.path === cleanPath || item.name === cleanPath);
  if (!folder) {
    const createdAt = nowIso();
    folder = {
      id: uniqueId("folder"),
      name: options.name || folderNameFromPath(cleanPath),
      path: cleanPath,
      fileCount: Number(options.fileCount || 0),
      createdAt,
      updatedAt: createdAt,
      chats: [],
    };
    state.workspace.folders.unshift(folder);
  } else {
    folder.name = options.name || folder.name;
    folder.fileCount = Number(options.fileCount || folder.fileCount || 0);
    folder.updatedAt = nowIso();
  }
  if (options.activate) {
    state.workspace.activeFolderId = folder.id;
    state.workspace.activeChatId = folder.chats[0]?.id || "";
  }
  if (options.save !== false) {
    saveWorkspace();
    renderFolderTree();
  }
  return folder;
}

function createWorkspaceChat(options = {}) {
  if (!state.workspace) loadWorkspace();
  let folder = activeFolder();
  if (!folder) folder = ensureWorkspaceFolder(contextValue("root", "samples"), { activate: true, save: false });
  if (!folder) return null;
  const createdAt = nowIso();
  const chat = {
    id: uniqueId("chat"),
    title: options.title || "New chat",
    createdAt,
    updatedAt: createdAt,
    messages: [],
  };
  folder.chats.unshift(chat);
  folder.updatedAt = createdAt;
  state.workspace.activeFolderId = folder.id;
  state.workspace.activeChatId = chat.id;
  if (options.render !== false) {
    saveWorkspace();
    renderFolderTree();
    renderActiveChat();
  }
  return chat;
}

function openWorkspaceChat(folderId, chatId) {
  if (!state.workspace) return;
  const folder = state.workspace.folders.find((item) => item.id === folderId);
  if (!folder) return;
  state.workspace.activeFolderId = folder.id;
  state.workspace.activeChatId = chatId || folder.chats[0]?.id || "";
  if (folder.path) upsertContext("root", "Root", folder.path, { syncFolder: false });
  saveWorkspace();
  renderFolderTree();
  renderActiveChat();
}

function deleteWorkspaceChat(folderId, chatId) {
  if (!state.workspace) return;
  const folder = state.workspace.folders.find((item) => item.id === folderId);
  if (!folder) return;
  const chat = folder.chats.find((item) => item.id === chatId);
  if (!chat) return;
  folder.chats = folder.chats.filter((item) => item.id !== chatId);
  folder.updatedAt = nowIso();
  if (state.workspace.activeFolderId === folderId && state.workspace.activeChatId === chatId) {
    state.workspace.activeChatId = folder.chats[0]?.id || "";
  }
  saveWorkspace();
  renderFolderTree();
  renderActiveChat();
}

function serializeAttachments(attachments = []) {
  return attachments.map((attachment) => ({
    kind: attachment.kind,
    source: attachment.source,
    folder: Boolean(attachment.folder),
    name: attachment.name,
    relativePath: attachment.relativePath,
    mime: attachment.mime,
    size: attachment.size,
  }));
}

function persistMessage(role, content, options = {}) {
  if (options.persist === false || state.restoringChat) return;
  if (!state.workspace) loadWorkspace();
  let chat = activeChat();
  if (!chat) chat = createWorkspaceChat({ render: false });
  if (!chat) return;
  const savedAt = nowIso();
  const attachments = serializeAttachments(options.attachments || []);
  const record = { role, content: String(content ?? ""), at: savedAt };
  if (attachments.length) record.attachments = attachments;
  chat.messages.push(record);
  chat.messages = chat.messages.slice(-200);
  if (role === "user" && (!chat.title || chat.title === "New chat")) {
    chat.title = chatTitleFromText(content || attachmentSummary(options.attachments || []));
  }
  chat.updatedAt = savedAt;
  const folder = activeFolder();
  if (folder) folder.updatedAt = savedAt;
  saveWorkspace();
  renderFolderTree();
}

function renderActiveChat() {
  const root = $("messages");
  if (!root) return;
  root.innerHTML = "";
  const chat = activeChat();
  if (!chat) {
    root.innerHTML = `
      <div class="chat-empty">
        <strong>No chat selected</strong>
        <span>Start a new chat from the left rail.</span>
      </div>
    `;
    return;
  }
  if (!chat.messages.length) {
    addMessage("system", "Axiom is online.", { persist: false });
    return;
  }
  state.restoringChat = true;
  try {
    chat.messages.forEach((message) => {
      addMessage(message.role || "assistant", message.content || "", {
        attachments: message.attachments || [],
        persist: false,
      });
    });
  } finally {
    state.restoringChat = false;
  }
}

function renderFolderTree() {
  const root = $("folderList");
  if (!root || !state.workspace) return;
  if (!state.workspace.folders.length) {
    root.innerHTML = `<div class="no-chats">No folders</div>`;
    return;
  }
  root.innerHTML = state.workspace.folders
    .map((folder) => {
      const active = folder.id === state.workspace.activeFolderId;
      const chats = folder.chats || [];
      return `
        <div class="folder-group ${active ? "active" : ""}">
          <button type="button" class="folder-row" data-folder-id="${escapeHtml(folder.id)}">
            <span class="folder-glyph"></span>
            <span class="folder-label">
              <strong>${escapeHtml(folder.name || "Workspace")}</strong>
              <small>${escapeHtml(folder.path || "Local folder")}${folder.fileCount ? ` · ${Number(folder.fileCount)} files` : ""}</small>
            </span>
          </button>
          <div class="chat-list">
            ${
              chats.length
                ? chats
                    .map(
                      (chat) => `
                        <div class="chat-row ${chat.id === state.workspace.activeChatId ? "active" : ""}">
                          <button type="button" class="chat-open" data-open-folder="${escapeHtml(folder.id)}" data-open-chat="${escapeHtml(chat.id)}">
                            <span class="chat-dot"></span>
                            <span>${escapeHtml(chat.title || "New chat")}</span>
                          </button>
                          <button type="button" class="delete-chat" title="Delete chat" aria-label="Delete ${escapeHtml(chat.title || "chat")}" data-delete-folder="${escapeHtml(folder.id)}" data-delete-chat="${escapeHtml(chat.id)}">x</button>
                        </div>
                      `,
                    )
                    .join("")
                : `<div class="no-chats">No chats</div>`
            }
          </div>
        </div>
      `;
    })
    .join("");
  root.querySelectorAll("[data-folder-id]").forEach((button) => {
    button.onclick = () => openWorkspaceChat(button.dataset.folderId, "");
  });
  root.querySelectorAll("[data-open-chat]").forEach((button) => {
    button.onclick = () => openWorkspaceChat(button.dataset.openFolder, button.dataset.openChat);
  });
  root.querySelectorAll("[data-delete-chat]").forEach((button) => {
    button.onclick = (event) => {
      event.stopPropagation();
      if (button.dataset.confirmDelete !== "true") {
        button.dataset.confirmDelete = "true";
        button.classList.add("confirming");
        button.textContent = "!";
        button.title = "Click again to delete";
        window.setTimeout(() => {
          if (!button.isConnected) return;
          button.dataset.confirmDelete = "false";
          button.classList.remove("confirming");
          button.textContent = "x";
          button.title = "Delete chat";
        }, 2200);
        return;
      }
      deleteWorkspaceChat(button.dataset.deleteFolder, button.dataset.deleteChat);
    };
  });
}

function handleWorkspaceFolderFiles(files) {
  if (!files.length) return;
  const firstPath = files[0].webkitRelativePath || files[0].name || "Selected folder";
  const folderName = firstPath.split(/[\\/]/).filter(Boolean)[0] || "Selected folder";
  ensureWorkspaceFolder(folderName, {
    name: folderName,
    fileCount: files.length,
    activate: true,
  });
  upsertContext("root", "Root", folderName, { syncFolder: false });
  renderActiveChat();
  addActivity("Folder selected", `${folderName} · ${files.length} files`);
}

const slashCommands = [
  { command: "/help", label: "Show command cards", detail: "Open a clickable guide to what Axiom can do." },
  { command: "/ask", label: "Ask with choices", detail: "Drop an interactive question card into chat." },
  { command: "/chat", label: "Chat with local model", detail: "Use Ollama for normal conversation without evidence retrieval." },
  { command: "/sources", label: "Show sources", detail: "Open indexed evidence sources." },
  { command: "/setup", label: "Check setup", detail: "Inspect local dependencies and readiness." },
  { command: "/mission", label: "SIH mission brief", detail: "Show NTRO problem coverage, gaps, and winning differentiators." },
  { command: "/image", label: "Image Lab", detail: "Open an image generation widget." },
  { command: "/vision", label: "Screen Vision", detail: "Capture or analyze a screenshot/image." },
  { command: "/graph", label: "Evidence graph", detail: "Open analytics, forecast, and timeline." },
  { command: "/investigate", label: "Investigation", detail: "Build a local dossier for a subject." },
  { command: "/report", label: "Report export", detail: "Generate a cited case report." },
  { command: "/context", label: "Manage context", detail: "Edit the active subject and folder root." },
  { command: "/clear", label: "Clear widgets", detail: "Close all open widgets." },
];

const quickPrompts = [
  { label: "Ask evidence", prompt: "international development 2024 screenshot" },
  { label: "Mission brief", prompt: "/mission" },
  { label: "Generate image", prompt: "generate image of an offline intelligence dashboard" },
  { label: "Investigate", prompt: "investigate international development" },
  { label: "Graph", prompt: "show analytics graph for international development" },
  { label: "Capture screen", prompt: "capture screenshot now" },
];

function addMessage(role, content, options = {}) {
  const root = $("messages");
  const item = document.createElement("div");
  item.className = `message ${role}`;
  const label = role === "user" ? "You" : role === "system" ? "System" : "Axiom";
  const avatar = role === "user" ? "Y" : "A";
  item.innerHTML = `
    <div class="message-meta">
      <span class="avatar">${avatar}</span>
      <span>${label}</span>
    </div>
    <div class="bubble"></div>
  `;
  const bubble = item.querySelector(".bubble");
  bubble.textContent = content;
  if (options.attachments?.length) {
    bubble.classList.add("has-attachments");
    const wrap = document.createElement("div");
    wrap.className = "message-attachments";
    options.attachments.forEach((attachment) => {
      const preview = document.createElement("div");
      preview.className = `message-attachment ${attachment.kind === "image" ? "" : "file-attachment"}`;
      if (attachment.kind === "image" && attachment.dataUrl) {
        preview.innerHTML = `
          <img alt="" src="${attachment.dataUrl}" />
          <span>${escapeHtml(attachment.name || "Pasted image")}</span>
        `;
      } else {
        preview.innerHTML = `
          <div class="file-glyph">${escapeHtml(fileExtension(attachment.name))}</div>
          <span>${escapeHtml(attachment.relativePath || attachment.name || "File")}</span>
        `;
      }
      wrap.appendChild(preview);
    });
    bubble.appendChild(wrap);
  }
  root.appendChild(item);
  root.scrollTop = root.scrollHeight;
  persistMessage(role, content, options);
  return item;
}

function addCardMessage(title, subtitle, bodyHtml, options = {}) {
  const root = $("messages");
  const item = document.createElement("div");
  item.className = "message widget-message assistant";
  item.innerHTML = `
    <div class="message-meta">
      <span class="avatar">A</span>
      <span>Axiom</span>
    </div>
    <div class="bubble">
      <div class="inline-card ${options.className || ""}">
        <div class="inline-card-head">
          <div>
            <p class="eyebrow">${escapeHtml(options.type || "Interactive")}</p>
            <h3>${escapeHtml(title)}</h3>
            ${subtitle ? `<span class="muted">${escapeHtml(subtitle)}</span>` : ""}
          </div>
          ${options.badge ? `<span class="badge ${options.badgeClass || ""}">${escapeHtml(options.badge)}</span>` : ""}
        </div>
        <div class="inline-card-body">${bodyHtml}</div>
      </div>
    </div>
  `;
  root.appendChild(item);
  root.scrollTop = root.scrollHeight;
  return item;
}

function addQuestionCard({ title, subtitle, options, allowOther = true, multiselect = false, onSubmit }) {
  const id = `question-${++state.questionSerial}`;
  const bodyHtml = `
    <div class="choice-grid" data-choice-grid>
      ${options
        .map(
          (option, index) => `
            <button type="button" class="choice-card" data-choice="${index}">
              <strong>${escapeHtml(option.label)}</strong>
              <span>${escapeHtml(option.description || "")}</span>
            </button>
          `,
        )
        .join("")}
    </div>
    ${
      allowOther
        ? `<div class="field"><label>Other</label><input data-other placeholder="Type a custom answer" /></div>`
        : ""
    }
    <div class="button-row">
      <button type="button" class="primary-button" data-submit>Use answer</button>
      <button type="button" class="ghost-button" data-dismiss>Dismiss</button>
    </div>
  `;
  const card = addCardMessage(title, subtitle, bodyHtml, { type: multiselect ? "Multi-select" : "Question", badge: "Click to answer", badgeClass: "blue" });
  card.dataset.questionId = id;
  const selected = new Set();
  card.querySelectorAll("[data-choice]").forEach((button) => {
    button.onclick = () => {
      const value = Number(button.dataset.choice);
      if (multiselect) {
        if (selected.has(value)) selected.delete(value);
        else selected.add(value);
      } else {
        selected.clear();
        selected.add(value);
      }
      card.querySelectorAll("[data-choice]").forEach((item) => item.classList.remove("selected"));
      selected.forEach((item) => card.querySelector(`[data-choice="${item}"]`)?.classList.add("selected"));
    };
  });
  card.querySelector("[data-dismiss]").onclick = () => {
    card.querySelector(".inline-card-body").innerHTML = `<span class="muted">Question dismissed.</span>`;
  };
  card.querySelector("[data-submit]").onclick = async () => {
    const answers = [...selected].map((index) => options[index]);
    const other = allowOther ? card.querySelector("[data-other]").value.trim() : "";
    if (!answers.length && !other) {
      card.querySelector("[data-other]")?.focus();
      return;
    }
    const payload = {
      answers,
      other,
      text: [...answers.map((item) => item.label), other].filter(Boolean).join(", "),
    };
    card.querySelector(".inline-card-body").innerHTML = `<span class="badge good">Answered</span><div class="approval-target">${escapeHtml(payload.text)}</div>`;
    addMessage("user", payload.text);
    if (onSubmit) await onSubmit(payload);
  };
  return card;
}

function addApprovalCard({ title, subtitle, target, confirmLabel = "Approve once", previewLabel = "Preview", onConfirm, onPreview }) {
  const bodyHtml = `
    <div class="approval-target">${escapeHtml(target)}</div>
    <div class="button-row">
      <button type="button" class="primary-button" data-confirm>${escapeHtml(confirmLabel)}</button>
      <button type="button" class="mini-button" data-preview>${escapeHtml(previewLabel)}</button>
      <button type="button" class="ghost-button" data-deny>Cancel</button>
    </div>
    <pre class="json-output compact" data-output></pre>
  `;
  const card = addCardMessage(title, subtitle, bodyHtml, {
    type: "Approval",
    badge: "Needs review",
    badgeClass: "warn",
    className: "approval-card",
  });
  const output = card.querySelector("[data-output]");
  card.querySelector("[data-deny]").onclick = () => {
    output.textContent = "Cancelled.";
    addActivity("Approval cancelled", title);
  };
  card.querySelector("[data-preview]").onclick = async () => {
    output.textContent = "Previewing...";
    try {
      output.textContent = jsonText(await onPreview());
    } catch (err) {
      output.textContent = err.message;
    }
  };
  card.querySelector("[data-confirm]").onclick = async () => {
    output.textContent = "Running approved action...";
    try {
      output.textContent = jsonText(await onConfirm());
      card.querySelector(".inline-card").classList.remove("approval-card");
      card.querySelector(".badge").textContent = "Approved";
      card.querySelector(".badge").className = "badge good";
    } catch (err) {
      output.textContent = err.message;
    }
  };
  return card;
}

function addActivity(title, detail = "") {
  state.activities.unshift({ title, detail, at: new Date() });
  state.activities = state.activities.slice(0, 6);
  renderActivity();
}

function renderActivity() {
  const root = $("activityList");
  if (!state.activities.length) {
    root.innerHTML = `<div class="activity-item"><strong>Idle</strong><span>Waiting</span></div>`;
    return;
  }
  root.innerHTML = state.activities
    .map(
      (item) => `
        <div class="activity-item">
          <strong>${escapeHtml(item.title)}</strong>
          <span>${escapeHtml(item.detail || item.at.toLocaleTimeString())}</span>
        </div>
      `,
    )
    .join("");
}

function renderContextChips() {
  const root = $("contextChips");
  if (!state.contextItems.length) {
    root.innerHTML = `<span class="context-chip"><strong>No context</strong><span>Add a subject or folder</span></span>`;
    return;
  }
  root.innerHTML = state.contextItems
    .map(
      (item) => `
        <span class="context-chip" title="${escapeHtml(item.value)}">
          <strong>${escapeHtml(item.label)}</strong>
          <span>${escapeHtml(item.value)}</span>
          <button type="button" data-remove-context="${escapeHtml(item.id)}">x</button>
        </span>
      `,
    )
    .join("");
  root.querySelectorAll("[data-remove-context]").forEach((button) => {
    button.onclick = () => {
      state.contextItems = state.contextItems.filter((item) => item.id !== button.dataset.removeContext);
      renderContextChips();
    };
  });
}

function upsertContext(type, label, value, options = {}) {
  if (!value) return;
  const existing = state.contextItems.find((item) => item.type === type);
  if (existing) {
    existing.label = label;
    existing.value = value;
  } else {
    state.contextItems.push({ id: `ctx-${type}-${Date.now()}`, type, label, value });
  }
  if (type === "subject") state.lastSubject = value;
  renderContextChips();
  if (type === "root" && options.syncFolder !== false) {
    ensureWorkspaceFolder(value, { activate: true });
  }
}

function contextValue(type, fallback = "") {
  return state.contextItems.find((item) => item.type === type)?.value || fallback;
}

function renderQuickActions() {
  $("quickActions").innerHTML = quickPrompts
    .map((item) => `<button type="button" class="quick-action" data-prompt="${escapeHtml(item.prompt)}">${escapeHtml(item.label)}</button>`)
    .join("");
  $("quickActions").querySelectorAll("[data-prompt]").forEach((button) => {
    button.onclick = () => runPrompt(button.dataset.prompt);
  });
}

function filesFromTransfer(dataTransfer, { imagesOnly = false } = {}) {
  const files = [];
  if (!dataTransfer) return files;
  if (dataTransfer.items?.length) {
    [...dataTransfer.items].forEach((item) => {
      if (item.kind === "file" && (!imagesOnly || item.type.startsWith("image/"))) {
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    });
  }
  if (!files.length && dataTransfer.files?.length) {
    [...dataTransfer.files].forEach((file) => {
      if (!imagesOnly || file.type.startsWith("image/")) files.push(file);
    });
  }
  return files;
}

async function addPendingFiles(files, { source = "file", folder = false } = {}) {
  const maxAttachments = 12;
  const maxBytes = 25 * 1024 * 1024;
  const slots = Math.max(0, maxAttachments - state.pendingAttachments.length);
  const accepted = files.slice(0, slots);
  if (!accepted.length) {
    addMessage("assistant", `Axiom can hold ${maxAttachments} staged items at a time.`);
    return;
  }

  for (const file of accepted) {
    if (file.size > maxBytes) {
      addMessage("assistant", `${file.name || "Selected file"} is larger than 25 MB, so I skipped it.`);
      continue;
    }
    const dataUrl = await readFileAsDataUrl(file);
    const id = `attachment-${++state.attachmentSerial}`;
    const extension = file.type.split("/")[1] || fileExtension(file.name) || "bin";
    const isImage = file.type.startsWith("image/");
    state.pendingAttachments.push({
      id,
      kind: isImage ? "image" : "file",
      source,
      folder,
      name: file.name || `${source}-${state.attachmentSerial}.${extension}`,
      relativePath: file.webkitRelativePath || file.name || `${source}-${state.attachmentSerial}.${extension}`,
      mime: file.type || "application/octet-stream",
      size: file.size,
      dataUrl,
    });
  }
  renderAttachmentTray();
  if (accepted.length) addActivity(folder ? "Folder ready" : source === "clipboard" ? "Clipboard image ready" : "Files ready", `${accepted.length} staged`);
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Could not read pasted image."));
    reader.readAsDataURL(file);
  });
}

function renderAttachmentTray() {
  const tray = $("attachmentTray");
  if (!state.pendingAttachments.length) {
    tray.hidden = true;
    tray.innerHTML = "";
    return;
  }
  tray.hidden = false;
  tray.innerHTML = state.pendingAttachments
    .map(
      (attachment) => `
        <div class="attachment-pill" data-attachment-id="${escapeHtml(attachment.id)}">
          ${
            attachment.kind === "image"
              ? `<img alt="" src="${attachment.dataUrl}" />`
              : `<div class="attachment-file-icon">${escapeHtml(fileExtension(attachment.name))}</div>`
          }
          <div>
            <strong>${escapeHtml(attachment.relativePath || attachment.name)}</strong>
            <span>${escapeHtml(attachment.sourceLabel || attachmentSourceLabel(attachment))} · ${escapeHtml(formatBytes(attachment.size))}</span>
          </div>
          <button type="button" title="Remove" data-remove-attachment="${escapeHtml(attachment.id)}">x</button>
        </div>
      `,
    )
    .join("");
  tray.querySelectorAll("[data-remove-attachment]").forEach((button) => {
    button.onclick = () => {
      state.pendingAttachments = state.pendingAttachments.filter((attachment) => attachment.id !== button.dataset.removeAttachment);
      renderAttachmentTray();
    };
  });
}

function formatBytes(bytes) {
  if (!bytes) return "clipboard image";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileExtension(name) {
  const clean = String(name || "").split(/[\\/]/).pop() || "";
  const match = clean.match(/\.([A-Za-z0-9]{1,5})$/);
  return match ? match[1].toUpperCase() : "FILE";
}

function attachmentSourceLabel(attachment) {
  if (attachment.source === "clipboard") return "Clipboard";
  if (attachment.folder) return "Folder";
  return attachment.kind === "image" ? "Image" : "File";
}

function evidenceToken(source) {
  const raw = String(source?.file_name || source?.file_path || "evidence").replaceAll('"', "").trim() || "evidence";
  return /\s/.test(raw) ? `@"${raw}"` : `@${raw}`;
}

function sourceByIndex(index) {
  return state.sources[Number(index)] || null;
}

function currentReferenceMatch() {
  const input = $("chatInput");
  if (!input) return null;
  const caret = input.selectionStart ?? input.value.length;
  const before = input.value.slice(0, caret);
  const match = before.match(/(^|\s)@([^\s@]*)$/);
  if (!match) return null;
  return {
    start: caret - match[0].length + match[1].length,
    end: caret,
    query: match[2] || "",
  };
}

function insertComposerText(text, options = {}) {
  const input = $("chatInput");
  if (!input) return;
  const match = options.replaceReference ? currentReferenceMatch() : null;
  const start = match ? match.start : input.selectionStart ?? input.value.length;
  const end = match ? match.end : input.selectionEnd ?? input.value.length;
  const before = input.value.slice(0, start);
  const after = input.value.slice(end);
  const leading = before && !/\s$/.test(before) ? " " : "";
  const trailing = after && !/^\s/.test(after) ? " " : "";
  const insert = `${leading}${text}${trailing}`;
  input.value = `${before}${insert}${after}`;
  const caret = before.length + insert.length;
  input.focus();
  input.setSelectionRange(caret, caret);
  resizeComposer();
  hideEvidenceMenu();
}

function insertEvidenceReference(source, options = {}) {
  if (!source) return;
  insertComposerText(`${evidenceToken(source)} `, { replaceReference: Boolean(options.replaceReference) });
  addActivity("Evidence referenced", source.file_name || source.file_path || "");
}

function toggleEvidenceDrawer(forceOpen) {
  const drawer = $("evidenceDrawer");
  const button = $("evidenceRailButton");
  if (!drawer || !button) return;
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : drawer.hidden;
  drawer.hidden = !shouldOpen;
  button.classList.toggle("active", shouldOpen);
  button.setAttribute("aria-expanded", String(shouldOpen));
  if (shouldOpen && !state.sources.length) refreshSources().catch((err) => addMessage("assistant", err.message));
}

function wireSourceRows(root) {
  root.querySelectorAll("[data-source-index]").forEach((row) => {
    row.onclick = () => insertEvidenceReference(sourceByIndex(row.dataset.sourceIndex));
    row.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        insertEvidenceReference(sourceByIndex(row.dataset.sourceIndex));
      }
    };
    row.ondragstart = (event) => {
      const source = sourceByIndex(row.dataset.sourceIndex);
      const token = evidenceToken(source);
      event.dataTransfer.setData("application/x-axiom-source", token);
      event.dataTransfer.setData("text/plain", token);
      event.dataTransfer.effectAllowed = "copy";
    };
  });
}

function ensureEvidenceMenu() {
  let menu = document.querySelector(".evidence-menu");
  if (menu) return menu;
  menu = document.createElement("div");
  menu.className = "evidence-menu";
  menu.hidden = true;
  document.body.appendChild(menu);
  return menu;
}

function hideEvidenceMenu() {
  const menu = ensureEvidenceMenu();
  menu.hidden = true;
}

function showEvidenceMenu(query = "") {
  const input = $("chatInput");
  const menu = ensureEvidenceMenu();
  const rect = input.getBoundingClientRect();
  menu.style.left = `${Math.max(12, rect.left)}px`;
  menu.style.top = `${Math.max(12, rect.top - 276)}px`;
  const cleanQuery = query.toLowerCase();
  const matches = state.sources
    .map((source, index) => ({ source, index }))
    .filter(({ source }) => {
      if (!cleanQuery) return true;
      return `${source.file_name || ""} ${source.file_path || ""} ${source.file_type || ""}`.toLowerCase().includes(cleanQuery);
    })
    .slice(0, 8);
  menu.innerHTML = `
    <div class="slash-menu-head">
      <p class="eyebrow">Reference evidence</p>
    </div>
    ${
      matches.length
        ? matches
            .map(
              ({ source, index }) => `
                <button type="button" class="slash-option evidence-option" data-evidence-index="${index}">
                  <strong>${escapeHtml(source.file_name || "Evidence")}</strong>
                  <span>${escapeHtml(evidenceToken(source))} · ${escapeHtml(source.file_type || "source")} · ${Number(source.chunk_count || 0)} chunks</span>
                </button>
              `,
            )
            .join("")
        : `<div class="menu-empty"><strong>No evidence</strong><span>Index files or refresh the Evidence drawer.</span></div>`
    }
  `;
  menu.hidden = false;
  menu.querySelectorAll("[data-evidence-index]").forEach((button) => {
    button.onclick = () => insertEvidenceReference(sourceByIndex(button.dataset.evidenceIndex), { replaceReference: true });
  });
}

function updateComposerMenus() {
  const input = $("chatInput");
  const value = input.value.trim();
  const showingSlash = value.startsWith("/");
  if (showingSlash) showSlashMenu();
  else hideSlashMenu();
  const match = currentReferenceMatch();
  if (!showingSlash && match) showEvidenceMenu(match.query);
  else hideEvidenceMenu();
}

async function handleAttachments(prompt, attachments) {
  const folderOrFiles = attachments.filter((attachment) => attachment.source !== "clipboard" || attachment.kind !== "image");
  const pastedImages = attachments.filter((attachment) => attachment.source === "clipboard" && attachment.kind === "image");
  if (folderOrFiles.length) await openUploadWidget(folderOrFiles, prompt);
  for (let index = 0; index < pastedImages.length; index += 1) {
    await openPastedImageWidget(pastedImages[index], prompt, index + 1, pastedImages.length);
  }
}

async function openPastedImageWidget(attachment, prompt, index, total) {
  const title = total > 1 ? `Pasted Snip ${index}` : "Pasted Snip";
  const { body } = createWidget(title, "Vision");
  body.innerHTML = `
    <div class="button-row">
      <span class="badge blue">Clipboard</span>
      <span class="badge">${escapeHtml(formatBytes(attachment.size))}</span>
      <span class="badge warn" data-save-state>Saving</span>
      <button type="button" class="mini-button" data-analyze disabled>Analyze</button>
    </div>
    <div class="vision-preview"><img data-preview class="ready" alt="" src="${attachment.dataUrl}" /></div>
    <pre class="json-output compact" data-output>Saving pasted image...</pre>
  `;
  const output = body.querySelector("[data-output]");
  const preview = body.querySelector("[data-preview]");
  const status = body.querySelector("[data-save-state]");
  const analyzeButton = body.querySelector("[data-analyze]");
  const data = await api("/api/vision/upload", {
    method: "POST",
    body: JSON.stringify({
      data_url: attachment.dataUrl,
      file_name: attachment.name,
      analyze: false,
      ingest: false,
    }),
  });
  output.textContent = jsonText(data);
  if (data.image?.image_path) {
    preview.src = `/api/artifact?path=${encodeURIComponent(data.image.image_path)}`;
    status.textContent = "Saved";
    status.className = "badge good";
    analyzeButton.disabled = false;
    analyzeButton.onclick = () => analyzeSavedPaste(body, data.image.image_path, prompt, true);
    if (shouldAutoAnalyzePaste(prompt)) analyzeSavedPaste(body, data.image.image_path, prompt, true);
  }
  addMessage("assistant", `Pasted image saved.${data.image?.image_path ? `\n${data.image.image_path}` : ""}`);
  addActivity("Pasted image saved", data.image?.image_path || attachment.name);
}

function shouldAutoAnalyzePaste(prompt) {
  return /\b(analy[sz]e|ocr|read|inspect|summari[sz]e|what is|what's|explain)\b/i.test(prompt || "");
}

async function analyzeSavedPaste(body, imagePath, prompt, ingest) {
  const output = body.querySelector("[data-output]");
  const analyzeButton = body.querySelector("[data-analyze]");
  const status = body.querySelector("[data-save-state]");
  analyzeButton.disabled = true;
  status.textContent = "Analyzing";
  status.className = "badge blue";
  output.textContent = "Analyzing saved image...";
  try {
    const data = await api("/api/vision/analyze", {
      method: "POST",
      body: JSON.stringify({
        image: imagePath,
        ingest,
        use_vlm: true,
        vision_model: "llama3.2-vision",
        prompt: prompt || null,
      }),
    });
    output.textContent = jsonText({ saved_image: imagePath, ...data });
    status.textContent = "Analyzed";
    status.className = "badge good";
    if (ingest) await refreshSources();
    addMessage("assistant", "Pasted image analysis is ready.");
    addActivity("Pasted image analyzed", imagePath);
  } catch (err) {
    output.textContent = err.message;
    status.textContent = "Needs setup";
    status.className = "badge warn";
    addActivity("Paste analysis paused", err.message);
  } finally {
    analyzeButton.disabled = false;
  }
}

async function openUploadWidget(attachments, prompt) {
  const hasFolder = attachments.some((attachment) => attachment.folder);
  const title = hasFolder ? "Folder Intake" : attachments.length === 1 ? "File Intake" : "Files Intake";
  const { body } = createWidget(title, "Evidence");
  body.innerHTML = `
    <div class="button-row">
      <span class="badge blue">${hasFolder ? "Folder" : "Files"}</span>
      <span class="badge">${attachments.length} item${attachments.length === 1 ? "" : "s"}</span>
      <span class="badge warn" data-upload-state>Importing</span>
    </div>
    <div class="result-list">
      ${attachments
        .slice(0, 8)
        .map(
          (attachment) => `
            <div class="result-item">
              <strong>${escapeHtml(attachment.relativePath || attachment.name)}</strong>
              <span class="muted">${escapeHtml(attachment.mime)} · ${escapeHtml(formatBytes(attachment.size))}</span>
            </div>
          `,
        )
        .join("")}
    </div>
    <pre class="json-output compact" data-output>Saving intake files...</pre>
  `;
  const output = body.querySelector("[data-output]");
  const status = body.querySelector("[data-upload-state]");
  const batchName = hasFolder ? folderBatchName(attachments) : undefined;
  const data = await api("/api/intake/upload", {
    method: "POST",
    body: JSON.stringify({
      batch_name: batchName,
      ingest: true,
      build_links: true,
      files: attachments.map((attachment) => ({
        name: attachment.name,
        relative_path: attachment.relativePath || attachment.name,
        mime_type: attachment.mime,
        data_url: attachment.dataUrl,
      })),
    }),
  });
  output.textContent = jsonText(data);
  status.textContent = "Indexed";
  status.className = "badge good";
  if (data.batch?.root_path) upsertContext("root", "Root", data.batch.root_path);
  await refreshSources();
  const chunkCount = Number(data.ingest?.chunks_created || 0);
  addMessage(
    "assistant",
    `${hasFolder ? "Folder" : "File"} intake saved and indexed.${data.batch?.root_path ? `\n${data.batch.root_path}` : ""}${chunkCount ? `\n${chunkCount} chunks ready.` : ""}`,
  );
  addActivity(hasFolder ? "Folder indexed" : "Files indexed", prompt || data.batch?.root_path || "");
}

function folderBatchName(attachments) {
  const first = attachments.find((attachment) => attachment.relativePath?.includes("/"))?.relativePath || "picked-folder";
  return first.split(/[\\/]/)[0] || "picked-folder";
}

function renderToolRun(container, { title, status = "Running", steps = [], detail = "" }) {
  container.innerHTML = `
    <div class="tool-run">
      <div class="tool-run-head">
        <strong>${escapeHtml(title)}</strong>
        <span class="badge ${status === "Done" ? "good" : status === "Failed" ? "bad" : "blue"}">${escapeHtml(status)}</span>
      </div>
      ${detail ? `<span class="muted">${escapeHtml(detail)}</span>` : ""}
      <div class="step-list">
        ${steps
          .map(
            (step) => `
              <div class="step-item ${escapeHtml(step.state || "")}">
                <span class="step-dot"></span>
                <div>
                  <strong>${escapeHtml(step.label)}</strong>
                  ${step.detail ? `<span class="muted">${escapeHtml(step.detail)}</span>` : ""}
                </div>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function clearWidgets() {
  $("widgetStack").innerHTML = `
    <div class="empty-widget">
      <strong>No widgets open</strong>
      <span>Results will appear here.</span>
    </div>
  `;
  $("widgetTitle").textContent = "Ready";
  closeSidePanel();
}

function createWidget(title, type = "Widget") {
  const stack = $("widgetStack");
  const empty = stack.querySelector(".empty-widget");
  if (empty) empty.remove();

  const id = `widget-${++state.widgetSerial}`;
  const widget = document.createElement("article");
  widget.className = "widget";
  widget.dataset.widgetId = id;
  widget.innerHTML = `
    <div class="widget-head">
      <div>
        <span class="widget-type">${escapeHtml(type)}</span>
        <h3>${escapeHtml(title)}</h3>
      </div>
      <button type="button" class="mini-button" data-close>Close</button>
    </div>
    <div class="widget-body"></div>
  `;
  widget.querySelector("[data-close]").onclick = () => {
    widget.remove();
    if (!stack.querySelector(".widget")) clearWidgets();
  };
  stack.prepend(widget);
  $("widgetTitle").textContent = title;
  openSidePanel();
  return {
    widget,
    body: widget.querySelector(".widget-body"),
    setTitle(nextTitle) {
      widget.querySelector("h3").textContent = nextTitle;
      $("widgetTitle").textContent = nextTitle;
    },
  };
}

function hasOpenWidgets() {
  return Boolean($("widgetStack")?.querySelector(".widget"));
}

function openSidePanel() {
  document.querySelector(".app-shell")?.classList.add("side-panel-open");
  $("sidePanelToggle")?.classList.add("active");
}

function closeSidePanel() {
  document.querySelector(".app-shell")?.classList.remove("side-panel-open");
  $("sidePanelToggle")?.classList.remove("active");
}

function toggleSidePanel() {
  const shell = document.querySelector(".app-shell");
  if (!shell) return;
  if (shell.classList.contains("side-panel-open")) {
    closeSidePanel();
    return;
  }
  if (hasOpenWidgets()) openSidePanel();
}

function ensureAttachMenu() {
  let menu = document.querySelector(".attach-menu");
  if (menu) return menu;
  menu = document.createElement("div");
  menu.className = "attach-menu";
  menu.hidden = true;
  document.body.appendChild(menu);
  return menu;
}

function showAttachMenu() {
  const button = $("attachButton");
  const menu = ensureAttachMenu();
  const rect = button.getBoundingClientRect();
  menu.style.left = `${Math.max(12, rect.left)}px`;
  menu.style.top = `${Math.max(12, rect.top - 226)}px`;
  menu.innerHTML = `
    <div class="slash-menu-head">
      <p class="eyebrow">Add to chat</p>
    </div>
    <button type="button" class="slash-option" data-attach-action="file">
      <strong>Choose files</strong>
      <span>Import documents, notes, images, or evidence into this thread.</span>
    </button>
    <button type="button" class="slash-option" data-attach-action="folder">
      <strong>Choose folder</strong>
      <span>Copy a picked folder into intake and index it.</span>
    </button>
    <button type="button" class="slash-option" data-attach-action="path">
      <strong>Use folder path</strong>
      <span>Work directly with a local folder by path, scan it, or index it.</span>
    </button>
    <button type="button" class="slash-option" data-attach-action="plugins">
      <strong>Plugins</strong>
      <span>Create images, research deeper, create tasks, or control the computer.</span>
    </button>
  `;
  menu.hidden = false;
  menu.querySelectorAll("[data-attach-action]").forEach((option) => {
    option.onclick = (event) => {
      event.stopPropagation();
      const action = option.dataset.attachAction;
      hideAttachMenu();
      if (action === "file") $("fileInput").click();
      if (action === "folder") $("folderInput").click();
      if (action === "path") openFolderPathWidget();
      if (action === "plugins") showPluginMenu(rect);
    };
  });
}

function showPluginMenu(anchorRect = $("attachButton").getBoundingClientRect()) {
  const menu = ensureAttachMenu();
  menu.style.left = `${Math.max(12, anchorRect.left)}px`;
  menu.style.top = `${Math.max(12, anchorRect.top - 282)}px`;
  menu.innerHTML = `
    <div class="slash-menu-head">
      <p class="eyebrow">Plugins</p>
    </div>
    <button type="button" class="slash-option" data-plugin-action="image">
      <strong>Create image</strong>
      <span>Open Image Lab for offline local image generation.</span>
    </button>
    <button type="button" class="slash-option" data-plugin-action="research">
      <strong>Deep research</strong>
      <span>Run a cited local dossier, graph, and report workflow.</span>
    </button>
    <button type="button" class="slash-option" data-plugin-action="task">
      <strong>Create task</strong>
      <span>Drop a structured task card into chat with next actions.</span>
    </button>
    <button type="button" class="slash-option" data-plugin-action="computer">
      <strong>Computer</strong>
      <span>Inspect windows, tabs, folders, files, and guarded commands.</span>
    </button>
    <button type="button" class="slash-option subtle-option" data-plugin-action="back">
      <strong>Back</strong>
      <span>Return to files and folder options.</span>
    </button>
  `;
  menu.hidden = false;
  menu.querySelectorAll("[data-plugin-action]").forEach((option) => {
    option.onclick = (event) => {
      event.stopPropagation();
      const action = option.dataset.pluginAction;
      if (action === "back") {
        showAttachMenu();
        return;
      }
      hideAttachMenu();
      runPluginAction(action);
    };
  });
}

function runPluginAction(action) {
  if (action === "image") return openImageWidget("offline intelligence evidence board with cited artifacts", false);
  if (action === "research") return openDeepResearchWidget();
  if (action === "task") return openTaskCreatorWidget();
  if (action === "computer") return openComputerControlWidget();
}

function hideAttachMenu() {
  ensureAttachMenu().hidden = true;
}

function openFolderPathWidget() {
  const { body } = createWidget("Folder Workspace", "Workspace");
  const currentRoot = contextValue("root", "samples");
  body.innerHTML = `
    <div class="field">
      <label>Folder path</label>
      <input data-folder-path value="${escapeHtml(currentRoot)}" placeholder="C:\\path\\to\\folder" />
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-index>Index folder</button>
      <button type="button" class="mini-button" data-scan>Scan</button>
      <button type="button" class="mini-button" data-use>Use as context</button>
    </div>
    <div class="widget-note">Picked folders are copied into intake. Folder paths work directly on local folders already on this laptop.</div>
  `;
  const pathValue = () => body.querySelector("[data-folder-path]").value.trim() || currentRoot;
  body.querySelector("[data-index]").onclick = () => runIngest(pathValue());
  body.querySelector("[data-scan]").onclick = () => runScan(pathValue());
  body.querySelector("[data-use]").onclick = () => {
    upsertContext("root", "Root", pathValue());
    addMessage("assistant", `Folder context set to ${pathValue()}.`);
    addActivity("Folder context", pathValue());
  };
}

function openDeepResearchWidget() {
  const { body } = createWidget("Deep Research", "Plugin");
  const subject = contextValue("subject", state.lastSubject);
  const root = contextValue("root", "samples");
  body.innerHTML = `
    <div class="widget-note">Offline deep research uses only indexed/local files, cited dossiers, analytics, and report export.</div>
    <div class="form-grid">
      <div class="field">
        <label>Subject</label>
        <input data-subject value="${escapeHtml(subject)}" />
      </div>
      <div class="field">
        <label>Folder root</label>
        <input data-root value="${escapeHtml(root)}" />
      </div>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-run>Run dossier</button>
      <button type="button" class="mini-button" data-graph>Graph</button>
      <button type="button" class="mini-button" data-report>Report</button>
    </div>
  `;
  const values = () => ({
    subject: body.querySelector("[data-subject]").value.trim() || subject,
    root: body.querySelector("[data-root]").value.trim() || root,
  });
  body.querySelector("[data-run]").onclick = () => {
    const next = values();
    runInvestigation(next.subject, [next.root].filter(Boolean));
  };
  body.querySelector("[data-graph]").onclick = () => runAnalytics(values().subject);
  body.querySelector("[data-report]").onclick = () => {
    const next = values();
    runReport(next.subject, [next.root].filter(Boolean));
  };
  addMessage("assistant", "Deep Research plugin is open.");
  addActivity("Plugin", "Deep Research");
}

function openTaskCreatorWidget() {
  const { body } = createWidget("Create Task", "Plugin");
  const subject = contextValue("subject", state.lastSubject);
  body.innerHTML = `
    <div class="form-grid">
      <div class="field">
        <label>Task title</label>
        <input data-title value="Review ${escapeHtml(subject)} evidence" />
      </div>
      <div class="field">
        <label>Mode</label>
        <select data-mode>
          <option value="research">Deep research</option>
          <option value="report">Report export</option>
          <option value="vision">Screen/image review</option>
          <option value="computer">Computer check</option>
        </select>
      </div>
      <div class="field span-all">
        <label>Instructions</label>
        <textarea data-instructions rows="3">Use local evidence only. Return citations, gaps, and next actions.</textarea>
      </div>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-card>Create task card</button>
      <button type="button" class="mini-button" data-start>Start now</button>
    </div>
  `;
  const payload = () => ({
    title: body.querySelector("[data-title]").value.trim() || "Untitled task",
    mode: body.querySelector("[data-mode]").value,
    instructions: body.querySelector("[data-instructions]").value.trim(),
  });
  body.querySelector("[data-card]").onclick = () => createTaskCard(payload());
  body.querySelector("[data-start]").onclick = () => startTask(payload());
  addMessage("assistant", "Task plugin is open.");
  addActivity("Plugin", "Create Task");
}

function createTaskCard(task) {
  const card = addCardMessage(
    task.title,
    task.instructions,
    `
      <div class="button-row">
        <span class="badge blue">${escapeHtml(task.mode)}</span>
        <button type="button" class="primary-button" data-start-task>Start</button>
        <button type="button" class="mini-button" data-edit-task>Edit in widget</button>
      </div>
    `,
    { type: "Task", badge: "Ready", badgeClass: "blue" },
  );
  card.querySelector("[data-start-task]").onclick = () => startTask(task);
  card.querySelector("[data-edit-task]").onclick = openTaskCreatorWidget;
  addActivity("Task created", task.title);
}

function startTask(task) {
  addMessage("user", task.title);
  if (task.mode === "research") return runInvestigation(contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
  if (task.mode === "report") return runReport(contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
  if (task.mode === "vision") return openVisionWidget({ mode: "capture", auto: false });
  if (task.mode === "computer") return openComputerControlWidget();
}

function openComputerControlWidget() {
  const { body } = createWidget("Computer", "Plugin");
  const root = contextValue("root", "samples");
  body.innerHTML = `
    <div class="widget-note">Computer actions stay local. Read-only checks run directly; commands and open actions keep their approval flow.</div>
    <div class="form-grid">
      <div class="field">
        <label>Folder</label>
        <input data-root value="${escapeHtml(root)}" />
      </div>
      <div class="field">
        <label>Find text</label>
        <input data-query value="screenshot" />
      </div>
      <div class="field span-all">
        <label>Command</label>
        <input data-command value="python --version" />
      </div>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-windows>Windows</button>
      <button type="button" class="mini-button" data-tabs>Tabs</button>
      <button type="button" class="mini-button" data-scan>Scan folder</button>
      <button type="button" class="mini-button" data-find>Find files</button>
      <button type="button" class="mini-button" data-command-check>Command</button>
    </div>
  `;
  const folder = () => body.querySelector("[data-root]").value.trim() || root;
  const query = () => body.querySelector("[data-query]").value.trim() || "screenshot";
  body.querySelector("[data-windows]").onclick = runWindows;
  body.querySelector("[data-tabs]").onclick = runTabs;
  body.querySelector("[data-scan]").onclick = () => runScan(folder());
  body.querySelector("[data-find]").onclick = () => runFind(folder(), query());
  body.querySelector("[data-command-check]").onclick = () => openCommandWidget(body.querySelector("[data-command]").value.trim() || "python --version", false);
  addMessage("assistant", "Computer plugin is open.");
  addActivity("Plugin", "Computer");
}

function ensureSlashMenu() {
  let menu = document.querySelector(".slash-menu");
  if (menu) return menu;
  menu = document.createElement("div");
  menu.className = "slash-menu";
  menu.hidden = true;
  document.body.appendChild(menu);
  return menu;
}

function showSlashMenu() {
  const input = $("chatInput");
  const value = input.value.trim().toLowerCase();
  const menu = ensureSlashMenu();
  const matches = slashCommands.filter((item) => item.command.startsWith(value || "/"));
  const rect = input.getBoundingClientRect();
  menu.style.left = `${Math.max(12, rect.left)}px`;
  menu.style.top = `${Math.max(12, rect.top - Math.min(360, matches.length * 58 + 46))}px`;
  menu.innerHTML = `
    <div class="slash-menu-head">
      <p class="eyebrow">Commands</p>
    </div>
    ${matches
      .map(
        (item) => `
          <button type="button" class="slash-option" data-command="${escapeHtml(item.command)}">
            <strong>${escapeHtml(item.command)} · ${escapeHtml(item.label)}</strong>
            <span>${escapeHtml(item.detail)}</span>
          </button>
        `,
      )
      .join("")}
  `;
  menu.hidden = !matches.length;
  menu.querySelectorAll("[data-command]").forEach((button) => {
    button.onclick = () => {
      input.value = button.dataset.command;
      hideSlashMenu();
      $("chatForm").requestSubmit();
    };
  });
}

function hideSlashMenu() {
  const menu = ensureSlashMenu();
  menu.hidden = true;
}

function setBusy(isBusy) {
  $("sendButton").disabled = isBusy;
}

function renderOutput(value, compact = false) {
  return `<pre class="json-output${compact ? " compact" : ""}">${escapeHtml(jsonText(value))}</pre>`;
}

async function handleSlashCommand(value) {
  const [command, ...rest] = value.trim().split(/\s+/);
  const args = rest.join(" ").trim();
  switch (command.toLowerCase()) {
    case "/help":
      return openHelpCard();
    case "/ask":
      return openQuestionBuilder();
    case "/chat":
      return runChat(args || "hi");
    case "/sources":
      return openSourcesWidget();
    case "/setup":
      return openSetupWidget();
    case "/mission":
      return openMissionWidget();
    case "/image":
      return openImageWidget(args || "offline intelligence command center dashboard", false);
    case "/vision":
      return openVisionWidget({ mode: "capture", auto: false });
    case "/graph":
      return runAnalytics(args || contextValue("subject", state.lastSubject));
    case "/investigate":
      return runInvestigation(args || contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
    case "/report":
      return runReport(args || contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
    case "/context":
      return openContextWidget();
    case "/clear":
      clearWidgets();
      addMessage("assistant", "Widgets cleared.");
      return;
    default:
      addMessage("assistant", `I do not know ${command} yet.`);
  }
}

function openHelpCard() {
  addQuestionCard({
    title: "What do you want Axiom to open?",
    subtitle: "Pick a workflow and I will open the right widget.",
    allowOther: false,
    options: [
      { label: "Evidence answer", description: "Ask indexed sources with citations.", prompt: contextValue("subject", state.lastSubject) },
      { label: "Image Lab", description: "Generate or configure local image output.", prompt: "/image" },
      { label: "Mission Brief", description: "Show SIH readiness and demo differentiators.", prompt: "/mission" },
      { label: "Investigation", description: "Build a dossier with evidence and guardrails.", prompt: `/investigate ${contextValue("subject", state.lastSubject)}` },
      { label: "Analytics", description: "Open graph, timeline, signals, and forecast.", prompt: `/graph ${contextValue("subject", state.lastSubject)}` },
      { label: "Screen Vision", description: "Capture or analyze a screenshot.", prompt: "/vision" },
      { label: "Sources", description: "Review indexed local evidence.", prompt: "/sources" },
    ],
    onSubmit: async ({ answers }) => {
      const prompt = answers[0]?.prompt;
      if (prompt) await runPrompt(prompt);
    },
  });
}

function openQuestionBuilder() {
  addQuestionCard({
    title: "Choose the next action",
    subtitle: "This is the clickable question widget pattern. The answer feeds back into chat.",
    allowOther: true,
    options: [
      { label: "Use evidence", description: "Answer from indexed sources." },
      { label: "Make a visual", description: "Open Image Lab." },
      { label: "Investigate", description: "Build a dossier." },
      { label: "Export report", description: "Create a case report." },
    ],
    onSubmit: async ({ text }) => {
      if (/visual|image/i.test(text)) return openImageWidget("offline intelligence dashboard", false);
      if (/investigate/i.test(text)) return runInvestigation(contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
      if (/report/i.test(text)) return runReport(contextValue("subject", state.lastSubject), [contextValue("root", "samples")].filter(Boolean));
      return runQuery(contextValue("subject", state.lastSubject));
    },
  });
}

function openContextWidget() {
  const { body } = createWidget("Context", "Workspace");
  body.innerHTML = `
    <div class="widget-note">These values bias slash commands and one-click workflows. They do not leave this local app.</div>
    <div class="form-grid">
      <div class="field">
        <label>Subject</label>
        <input data-subject value="${escapeHtml(contextValue("subject", state.lastSubject))}" />
      </div>
      <div class="field">
        <label>Folder root</label>
        <input data-root value="${escapeHtml(contextValue("root", "samples"))}" />
      </div>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-save>Save context</button>
      <button type="button" class="mini-button" data-question>Ask me instead</button>
    </div>
  `;
  body.querySelector("[data-save]").onclick = () => {
    upsertContext("subject", "Subject", body.querySelector("[data-subject]").value.trim());
    upsertContext("root", "Root", body.querySelector("[data-root]").value.trim());
    addMessage("assistant", "Context updated.");
    addActivity("Context updated", contextValue("subject", state.lastSubject));
  };
  body.querySelector("[data-question]").onclick = () => {
    addQuestionCard({
      title: "Set active context",
      subtitle: "Choose what Axiom should focus on next.",
      options: [
        { label: "International development", description: "Use the demo evidence subject.", subject: "international development" },
        { label: "Jordan Vale", description: "Use the investigation example subject.", subject: "Jordan Vale" },
        { label: "Screenshot evidence", description: "Focus queries on screenshots.", subject: "screenshot evidence" },
      ],
      onSubmit: async ({ answers, other }) => {
        upsertContext("subject", "Subject", other || answers[0]?.subject || answers[0]?.label);
        addMessage("assistant", "Context updated from your choice.");
      },
    });
  };
  addActivity("Context opened", "Subject and root");
}

async function refreshSetup() {
  const data = await api("/api/setup/status");
  state.setup = data;
  $("dbPath").textContent = data.executable ? data.executable.split(/[\\/]/).pop() : "Local runtime";
  $("dbPath").title = data.executable || "Local runtime";
  $("readyText").textContent = data.ready ? "Full mode ready" : `${data.required_installed}/${data.required_total} ready`;
  $("readyDot").classList.toggle("ready", Boolean(data.ready));
  return data;
}

async function refreshSources() {
  const data = await api("/api/sources");
  state.sources = data.sources || [];
  renderSourcesRail();
  return state.sources;
}

function renderSourcesRail() {
  const list = $("sourcesList");
  const count = $("evidenceCount");
  if (count) count.textContent = String(state.sources.length);
  if (!state.sources.length) {
    list.innerHTML = `<div class="source-item empty-source"><strong>No evidence</strong><small>Index a folder to begin.</small></div>`;
    return;
  }
  list.innerHTML = state.sources
    .slice(0, 24)
    .map(
      (source, index) => `
        <div class="source-item" role="button" tabindex="0" draggable="true" data-source-index="${index}" title="Click or drag to reference this evidence">
          <strong>${escapeHtml(source.file_name)}</strong>
          <small>${escapeHtml(source.file_type)} · ${escapeHtml(source.status)} · ${Number(source.chunk_count || 0)} chunks</small>
          <span class="source-reference">${escapeHtml(evidenceToken(source))}</span>
        </div>
      `,
    )
    .join("");
  wireSourceRows(list);
}

function detectIntent(raw) {
  const text = raw.trim();
  const lower = text.toLowerCase();
  if (isSmallTalk(lower)) {
    return { kind: "chat", message: text };
  }
  const wantsScreenCapture =
    /\b(take|capture|grab)\b.*\b(screenshot|screen|active window)\b/.test(lower) ||
    /\b(screenshot|screen capture)\s+(now|please)\b/.test(lower) ||
    /\blook at (my|the) screen\b/.test(lower) ||
    /\bactive window\b/.test(lower);

  if (/\b(setup|dependency|dependencies|install missing|health|ready|status)\b/.test(lower)) {
    return { kind: "setup" };
  }
  if (/\b(sih|ntro|mission brief|readiness|winning differentiators|problem statement|demo brief)\b/.test(lower)) {
    return { kind: "mission" };
  }
  if (/\b(sources|indexed evidence|source list)\b/.test(lower)) {
    return { kind: "sources" };
  }
  if (/\b(ask card|question card|choices|pick one|options)\b/.test(lower)) {
    return { kind: "question" };
  }
  if (/\b(report|case report|export report)\b/.test(lower)) {
    return { kind: "report", subject: extractSubject(text), roots: extractRoots(text) };
  }
  if (/\b(investigate|dossier|case file|profile)\b/.test(lower)) {
    return { kind: "investigate", subject: extractSubject(text), roots: extractRoots(text) };
  }
  if (/\.(png|jpe?g|webp|bmp|gif)\b/i.test(text) && /\b(analy[sz]e|ocr|vision|read|inspect)\b/.test(lower)) {
    return { kind: "analyze-image", imagePath: extractImagePath(text) };
  }
  if (wantsScreenCapture) {
    return { kind: "capture", activeWindow: /\bactive window\b/.test(lower) };
  }
  if (/\b(image|picture|photo|poster|visual|draw|render|stable diffusion|diffusion)\b/.test(lower)) {
    return { kind: "image", prompt: cleanImagePrompt(text) };
  }
  if (/\b(analytics|evidence graph|graph|timeline|prediction|predict|forecast|signals)\b/.test(lower)) {
    return { kind: "analytics", query: cleanAnalyticsQuery(text) };
  }
  if (/\b(ingest|index|import|add evidence)\b/.test(lower)) {
    return { kind: "ingest", path: extractPath(text, "samples") };
  }
  if (/\b(browser tabs|tabs|current tabs)\b/.test(lower)) {
    return { kind: "tabs" };
  }
  if (/\b(windows|visible apps|open apps)\b/.test(lower)) {
    return { kind: "windows" };
  }
  if (/^open\s+/i.test(text)) {
    return { kind: "open-path", path: text.replace(/^open\s+/i, "").trim() };
  }
  if (/\b(find|search)\b/.test(lower) && /\b(file|files|folder|path|in|under|content)\b/.test(lower)) {
    const parsed = extractFind(text);
    return { kind: "find", query: parsed.query, path: parsed.path };
  }
  if (/\b(scan|list files|map folder|folder map)\b/.test(lower)) {
    return { kind: "scan", path: extractPath(text, ".") };
  }
  if (/^(run|command|terminal|execute)\b/i.test(text)) {
    return {
      kind: "command",
      command: extractCommand(text),
      execute: !/\b(dry run|preview|do not execute)\b/.test(lower),
    };
  }
  if (/^chat\s+/i.test(text)) {
    return { kind: "chat", message: text.replace(/^chat\s+/i, "").trim() || text };
  }
  return { kind: "query", query: text };
}

function isSmallTalk(lower) {
  const clean = lower.replace(/[!.?,\s]+$/g, "").trim();
  return /^(hi|hey|hello|yo|sup|thanks|thank you|ty|ok|okay|cool|nice|what can you do|help me|start)$/.test(clean);
}

function cleanImagePrompt(text) {
  const cleaned = text
    .replace(/^\s*(please\s+)?(generate|make|create|draw|render)\s+(me\s+)?(an?\s+)?(image|picture|photo|poster|visual)\s*(of|for|about)?\s*/i, "")
    .replace(/^\s*(image|picture|photo|poster|visual)\s*(of|for|about)?\s*/i, "")
    .trim();
  return cleaned || text;
}

function cleanAnalyticsQuery(text) {
  const cleaned = text
    .replace(/\b(show|build|make|create|open|run)\b/gi, "")
    .replace(/\b(analytics|evidence graph|graph|timeline|prediction|predict|forecast|signals)\b/gi, "")
    .replace(/\b(for|about|on)\b/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || state.lastSubject;
}

function extractSubject(text) {
  const match = text.match(/\b(?:investigate|dossier|report|case report|case file|profile)\s*(?:on|for|about)?\s+(.+)$/i);
  const raw = (match ? match[1] : text).replace(/\s+(?:in|under|from|root|with roots?)\s+.+$/i, "").trim();
  const subject = raw || state.lastSubject;
  state.lastSubject = subject;
  upsertContext("subject", "Subject", subject);
  return subject;
}

function extractRoots(text) {
  const root = extractPath(text, "");
  return root ? [root] : [];
}

function extractPath(text, fallback) {
  const quoted = text.match(/"([^"]+)"|'([^']+)'/);
  if (quoted) return cleanPath(quoted[1] || quoted[2]);

  const winPath = text.match(/[A-Za-z]:\\[^\n]+/);
  if (winPath) return cleanPath(winPath[0]);

  const relPath = text.match(/(?:^|\s)(\.{1,2}(?:[\\/][^\s]+)?|samples|artifacts|exports|data|docs|intake)(?=$|\s)/i);
  if (relPath) return cleanPath(relPath[1]);

  const after = text.match(/\b(?:in|under|from|folder|path|root|scan|index|ingest|import)\s+(.+)$/i);
  if (after) return cleanPath(after[1].split(/\s+(?:for|about|with)\s+/i)[0]);

  return fallback;
}

function cleanPath(value) {
  return String(value || "")
    .replace(/^folder\s+/i, "")
    .replace(/[.,;]+$/g, "")
    .trim();
}

function extractImagePath(text) {
  const quoted = text.match(/"([^"]+\.(?:png|jpe?g|webp|bmp|gif))"|'([^']+\.(?:png|jpe?g|webp|bmp|gif))'/i);
  if (quoted) return cleanPath(quoted[1] || quoted[2]);
  const winPath = text.match(/[A-Za-z]:\\[^\n]+\.(?:png|jpe?g|webp|bmp|gif)/i);
  if (winPath) return cleanPath(winPath[0]);
  const relPath = text.match(/(?:\.{1,2}[\\/])?[^\s]+\.(?:png|jpe?g|webp|bmp|gif)/i);
  return relPath ? cleanPath(relPath[0]) : "";
}

function extractFind(text) {
  const match = text.match(/\b(?:find|search)\s+(.+?)\s+(?:in|under|from)\s+(.+)$/i);
  if (match) {
    return {
      query: cleanFindQuery(match[1]),
      path: cleanPath(match[2]),
    };
  }
  return {
    query: cleanFindQuery(text.replace(/\b(find|search|files|content)\b/gi, "")),
    path: extractPath(text, "."),
  };
}

function cleanFindQuery(value) {
  return String(value || "screenshot").replace(/^for\s+/i, "").replace(/["']/g, "").trim() || "screenshot";
}

function extractCommand(text) {
  const command = text.replace(/^\s*(run|command|terminal|execute)\s*[:\-]?\s*/i, "").trim();
  return command || "python --version";
}

async function handleIntent(intent, raw) {
  switch (intent.kind) {
    case "chat":
      return runChat(intent.message || raw);
    case "question":
      return openQuestionBuilder();
    case "setup":
      return openSetupWidget();
    case "mission":
      return openMissionWidget();
    case "sources":
      return openSourcesWidget();
    case "image":
      return openImageWidget(intent.prompt, true);
    case "capture":
      return openVisionWidget({ mode: "capture", activeWindow: intent.activeWindow, auto: true });
    case "analyze-image":
      return openVisionWidget({ mode: "analyze", imagePath: intent.imagePath, auto: true });
    case "ingest":
      return runIngest(intent.path);
    case "investigate":
      return runInvestigation(intent.subject, intent.roots);
    case "report":
      return runReport(intent.subject, intent.roots);
    case "analytics":
      return runAnalytics(intent.query);
    case "windows":
      return runWindows();
    case "tabs":
      return runTabs();
    case "scan":
      return runScan(intent.path);
    case "find":
      return runFind(intent.path, intent.query);
    case "open-path":
      return runOpenPath(intent.path);
    case "command":
      return openCommandWidget(intent.command, intent.execute);
    case "query":
    default:
      return runQuery(raw || intent.query);
  }
}

async function openSetupWidget() {
  const { body } = createWidget("Setup", "System");
  body.innerHTML = renderOutput("Checking local dependencies...", true);
  const data = await refreshSetup();
  renderSetup(body, data);
  const missing = data.required_total - data.required_installed;
  addMessage("assistant", data.ready ? "Full mode is ready." : `${missing} required setup item${missing === 1 ? "" : "s"} still need attention.`);
  addActivity("Setup checked", data.ready ? "Ready" : `${data.required_installed}/${data.required_total}`);
}

async function openMissionWidget() {
  const { body } = createWidget("SIH25231 Mission Brief", "NTRO");
  renderToolRun(body, {
    title: "Building mission brief",
    status: "Running",
    detail: "Checking corpus coverage, offline adapters, and differentiators",
    steps: [
      { label: "Read evidence ledger", state: "running" },
      { label: "Score multimodal coverage", state: "" },
      { label: "Prepare demo battlecard", state: "" },
    ],
  });
  const data = await api("/api/mission/brief");
  body.innerHTML = `
    <div class="mission-hero">
      <div>
        <span class="widget-type">${escapeHtml(data.problem.ps_id)} · ${escapeHtml(data.problem.organization)}</span>
        <h3>${escapeHtml(data.verdict)}</h3>
        <span class="muted">${escapeHtml(data.problem.statement)}</span>
      </div>
      <div class="mission-score">${Number(data.score || 0)}</div>
    </div>
    <div class="mission-grid">
      ${(data.coverage || [])
        .map(
          (item) => `
            <div class="mission-tile ${item.ready ? "ready" : "gap"}">
              <strong>${escapeHtml(item.label)}</strong>
              <span>${item.ready ? "Ready" : "Gap"} · ${Number(item.indexed || 0)} indexed · ${Number(item.needs_adapter || 0)} adapter notes</span>
            </div>
          `,
        )
        .join("")}
    </div>
    <div class="split-two">
      <div>
        <span class="mini-label">Differentiators</span>
        <div class="cards">
          ${(data.differentiators || [])
            .map((item) => `<div class="cardlet"><strong>${escapeHtml(item.title)}</strong><span>${escapeHtml(item.body)}</span></div>`)
            .join("")}
        </div>
      </div>
      <div>
        <span class="mini-label">Demo Script</span>
        <div class="timeline">
          ${(data.demo_script || [])
            .map((item, index) => `<div class="timeline-item"><strong>Step ${index + 1}</strong><span>${escapeHtml(item)}</span></div>`)
            .join("")}
        </div>
      </div>
    </div>
    <div>
      <span class="mini-label">Corpus</span>
      <div class="button-row">
        <span class="badge">${Number(data.corpus?.documents || 0)} docs</span>
        <span class="badge">${Number(data.corpus?.chunks || 0)} chunks</span>
        <span class="badge">${Number(data.corpus?.cross_modal_links || 0)} links</span>
        <span class="badge">${Number(data.corpus?.audited_actions || 0)} audit actions</span>
      </div>
    </div>
    ${
      data.gaps?.length
        ? `<div><span class="mini-label">Win Gaps</span><div class="cards">${data.gaps
            .map((gap) => `<div class="cardlet"><span>${escapeHtml(gap)}</span></div>`)
            .join("")}</div></div>`
        : `<span class="badge good">No critical SIH gaps detected</span>`
    }
    <details>
      <summary>Raw mission brief</summary>
      ${renderOutput(data, true)}
    </details>
  `;
  addMessage("assistant", `Mission brief is open. SIH readiness score: ${Number(data.score || 0)}/100.`);
  addActivity("Mission brief", `${Number(data.score || 0)}/100`);
}

function renderSetup(body, data) {
  body.innerHTML = `
    <div class="button-row">
      <span class="badge ${data.ready ? "good" : "warn"}">${data.ready ? "Ready" : `${data.required_installed}/${data.required_total} ready`}</span>
      <button type="button" class="mini-button" data-refresh>Refresh</button>
      <button type="button" class="mini-button" data-install>Install missing</button>
    </div>
    <div class="dependency-list">
      ${(data.checks || [])
        .map(
          (dep) => `
            <div class="dependency-item">
              <div class="dependency-main">
                <strong>${escapeHtml(dep.label)}</strong>
                <span class="muted">${escapeHtml(dep.category)} · ${escapeHtml(dep.note)}</span>
              </div>
              <div class="dependency-actions">
                <span class="badge ${dep.installed ? "good" : dep.required ? "bad" : "warn"}">${dep.installed ? "ready" : "missing"}</span>
                ${
                  !dep.installed && dep.install_plan?.length
                    ? `<button type="button" class="mini-button dependency-install" data-install-one="${escapeHtml(dep.key)}">Install</button>`
                    : ""
                }
                <span class="dependency-status" data-install-status="${escapeHtml(dep.key)}"></span>
              </div>
            </div>
          `,
        )
        .join("")}
    </div>
    <pre class="json-output compact" data-log></pre>
  `;
  body.querySelector("[data-refresh]").onclick = async () => renderSetup(body, await refreshSetup());
  body.querySelector("[data-install]").onclick = async () => {
    if (!state.setup) await refreshSetup();
    const missing = state.setup.checks.filter((dep) => dep.required && !dep.installed && dep.install_plan.length);
    const names = missing.map((dep) => dep.label).join(", ") || "No installable required dependencies";
    addApprovalCard({
      title: "Install missing dependencies?",
      subtitle: "This may download packages or models depending on each dependency plan.",
      target: names,
      confirmLabel: "Install once",
      previewLabel: "Show plan",
      onPreview: () => ({ missing }),
      onConfirm: () => installMissingRequired(body.querySelector("[data-log]")),
    });
  };
  body.querySelectorAll("[data-install-one]").forEach((button) => {
    button.onclick = () => installSingleDependency(button.dataset.installOne, body);
  });
}

async function installSingleDependency(key, body) {
  if (!state.setup) await refreshSetup();
  const dep = state.setup.checks.find((item) => item.key === key);
  const log = body.querySelector("[data-log]");
  if (!dep || dep.installed || !dep.install_plan.length) {
    log.textContent = "Nothing installable for this setup item.";
    return { success: false, note: "Nothing installable for this setup item." };
  }
  const button = [...body.querySelectorAll("[data-install-one]")].find((item) => item.dataset.installOne === key);
  const status = [...body.querySelectorAll("[data-install-status]")].find((item) => item.dataset.installStatus === key);
  if (button) button.disabled = true;
  if (button) button.textContent = "Installing";
  if (status) {
    status.textContent = "running";
    status.className = "dependency-status running";
  }
  log.textContent = installProgressText(dep, 0);
  log.scrollIntoView({ block: "nearest" });
  try {
    const result = await runDependencyInstall(dep, {
      log,
      status,
      onTick: (elapsed) => {
        log.textContent = installProgressText(dep, elapsed);
        if (status) status.textContent = dep.category === "model" ? `pulling ${elapsed}s` : `running ${elapsed}s`;
      },
    });
    const data = await refreshSetup();
    renderSetup(body, data);
    const nextDep = state.setup?.checks.find((item) => item.key === key);
    const nextStatus = [...body.querySelectorAll("[data-install-status]")].find((item) => item.dataset.installStatus === key);
    if (nextStatus) {
      nextStatus.textContent = result.success && nextDep?.installed ? "installed" : result.success ? "installed, refresh app" : "failed";
      nextStatus.className = `dependency-status ${result.success ? "done" : "failed"}`;
    }
    const nextLog = body.querySelector("[data-log]");
    nextLog.textContent = `${dep.label}: ${result.success ? "ok" : "failed"}\n${result.stdout || result.stderr || result.note || ""}`;
    addActivity("Setup install", `${dep.label}: ${result.success ? "ok" : "failed"}`);
    return result;
  } catch (err) {
    log.textContent = `${dep.label}: failed\n${err.message}`;
    if (status) {
      status.textContent = "failed";
      status.className = "dependency-status failed";
    }
    if (button) button.disabled = false;
    if (button) button.textContent = "Install";
    addActivity("Setup install failed", dep.label);
    return { success: false, error: err.message };
  }
}

function installProgressText(dep, elapsedSeconds) {
  if (dep.category === "model") {
    return `Pulling ${dep.label} with Ollama...\nThis can take several minutes for the first download.\nElapsed: ${elapsedSeconds}s`;
  }
  if (dep.category === "system") {
    return `Installing ${dep.label}...\nA Windows installer may open or run in the background.\nElapsed: ${elapsedSeconds}s`;
  }
  return `Installing ${dep.label}...\nElapsed: ${elapsedSeconds}s`;
}

async function runDependencyInstall(dep, { log, status, onTick } = {}) {
  const startedAt = Date.now();
  const job = await api("/api/setup/install-start", {
    method: "POST",
    body: JSON.stringify({ key: dep.key }),
  });
  const jobId = job.job_id;
  if (!jobId) throw new Error("Install did not return a job id.");
  while (true) {
    const elapsed = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
    if (onTick) onTick(elapsed);
    const current = await api(`/api/setup/install-status?job_id=${encodeURIComponent(jobId)}`);
    if (current.status === "done" || current.status === "failed") {
      return current.result || { success: current.status === "done", note: "Installer finished." };
    }
    if (log && elapsed % 10 === 0) log.textContent = installProgressText(dep, elapsed);
    if (status && elapsed > 0) status.className = "dependency-status running";
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function installMissingRequired(log) {
  if (!state.setup) await refreshSetup();
  const missing = state.setup.checks.filter((dep) => dep.required && !dep.installed && dep.install_plan.length);
  if (!missing.length) {
    log.textContent = "All installable required dependencies are ready.";
    return { success: true, note: "All installable required dependencies are ready." };
  }
  const rows = [];
  for (const dep of missing) {
    log.textContent = `${installProgressText(dep, 0)}\n\n${rows.join("\n\n")}`;
    const result = await runDependencyInstall(dep, {
      log,
      onTick: (elapsed) => {
        log.textContent = `${installProgressText(dep, elapsed)}\n\n${rows.join("\n\n")}`;
      },
    });
    rows.push(`${dep.label}: ${result.success ? "ok" : "failed"}\n${result.stdout || result.stderr || result.note || ""}`);
    log.textContent = rows.join("\n\n");
  }
  await refreshSetup();
  return { success: true, installed: rows };
}

async function openSourcesWidget() {
  const { body } = createWidget("Sources", "Evidence");
  body.innerHTML = renderOutput("Refreshing indexed sources...", true);
  const sources = await refreshSources();
  body.innerHTML = renderSourcesList(sources);
  addMessage("assistant", `${sources.length} source${sources.length === 1 ? "" : "s"} are indexed.`);
  addActivity("Sources refreshed", `${sources.length} indexed`);
}

function renderSourcesList(sources) {
  if (!sources.length) {
    return `<div class="result-item"><strong>No sources indexed</strong><span class="muted">Index a folder to begin.</span></div>`;
  }
  return `
    <div class="result-list">
      ${sources
        .map(
          (source) => `
            <div class="result-item">
              <strong>${escapeHtml(source.file_name)}</strong>
              <span class="muted">${escapeHtml(source.file_type)} · ${escapeHtml(source.status)} · ${Number(source.chunk_count || 0)} chunks</span>
              <span class="muted">${escapeHtml(source.file_path)}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

async function runQuery(query) {
  const { body } = createWidget("Evidence Answer", "HiveRAG");
  renderToolRun(body, {
    title: "HiveRAG evidence run",
    status: "Running",
    detail: query,
    steps: [
      { label: "Route sphere memory", state: "done" },
      { label: "Grow tree and hex paths", state: "running" },
      { label: "Walk evidence web", state: "" },
      { label: "Validate leaf citations", state: "" },
    ],
  });
  const data = await api("/api/query", {
    method: "POST",
    body: JSON.stringify({ query, top_k: 5 }),
  });
  body.innerHTML = `
    <div class="answer-block">${escapeHtml(data.answer)}</div>
    <div>
      <span class="mini-label">Sources</span>
      ${renderEvidenceSources(data.sources || [])}
    </div>
  `;
  addMessage("assistant", "I found an evidence-backed answer and opened the cited sources.");
  addActivity("Query answered", query);
}

async function runChat(message) {
  const text = String(message || "").trim();
  if (!text) return;
  addActivity("Local chat", "Thinking");
  try {
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message: text,
        history: chatHistoryForModel(),
      }),
    });
    addMessage("assistant", data.response || "The local model did not return a response.");
    addActivity("Local chat", data.used_model ? data.model || "Ollama" : "Fallback");
  } catch (err) {
    addMessage("assistant", `Local chat is not responding yet: ${err.message}`);
    addActivity("Local chat failed", err.message);
  }
}

function chatHistoryForModel() {
  const chat = activeChat();
  if (!chat?.messages?.length) return [];
  return chat.messages
    .slice(-13, -1)
    .filter((item) => item.role === "user" || item.role === "assistant")
    .map((item) => ({
      role: item.role,
      content: String(item.content || "").slice(0, 2000),
    }));
}

function renderEvidenceSources(sources) {
  if (!sources.length) {
    return `<div class="result-item"><strong>No cited sources</strong><span class="muted">Try indexing more evidence.</span></div>`;
  }
  return `
    <div class="result-list">
      ${sources
        .map(
          (source) => `
            <div class="result-item">
              <strong>${escapeHtml(source.citation)} ${escapeHtml(source.file_name)}</strong>
              <span class="muted">${escapeHtml(source.location || "")}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

async function runIngest(path) {
  upsertContext("root", "Root", path);
  const { body } = createWidget("Index Evidence", "Ingest");
  renderToolRun(body, {
    title: "Index evidence",
    status: "Running",
    detail: path,
    steps: [
      { label: "Scan folder", state: "running" },
      { label: "Extract text", state: "" },
      { label: "Build chunks and links", state: "" },
    ],
  });
  const data = await api("/api/ingest", {
    method: "POST",
    body: JSON.stringify({ path, build_links: true }),
  });
  body.innerHTML = `
    <div class="button-row">
      <span class="badge good">${Number(data.chunks_created || 0)} chunks</span>
      <span class="badge">${(data.indexed_files || []).length} files</span>
    </div>
    ${renderOutput(data)}
  `;
  await refreshSources();
  addMessage("assistant", `Indexed ${path}.`);
  addActivity("Evidence indexed", path);
}

async function openImageWidget(prompt, autoRun = false) {
  const { body } = createWidget("Image Lab", "Image");
  body.innerHTML = `
    <div class="form-grid">
      <div class="field span-all">
        <label>Prompt</label>
        <textarea data-prompt rows="4"></textarea>
      </div>
      <div class="field span-all">
        <label>Negative prompt</label>
        <textarea data-negative rows="2">blurry, low quality, distorted text, watermark</textarea>
      </div>
      <div class="field">
        <label>Backend</label>
        <select data-backend>
          <option value="auto">Auto</option>
          <option value="a1111">AUTOMATIC1111 / Forge</option>
          <option value="diffusers">Diffusers</option>
        </select>
      </div>
      <div class="field">
        <label>Model path</label>
        <input data-model-path placeholder="C:\\models\\stable-diffusion-local" />
      </div>
      <div class="field">
        <label>Width</label>
        <input data-width type="number" min="256" max="1536" step="64" value="768" />
      </div>
      <div class="field">
        <label>Height</label>
        <input data-height type="number" min="256" max="1536" step="64" value="512" />
      </div>
      <div class="field">
        <label>Steps</label>
        <input data-steps type="number" min="1" max="80" value="24" />
      </div>
      <div class="field">
        <label>CFG</label>
        <input data-cfg type="number" min="1" max="20" step="0.5" value="7" />
      </div>
      <div class="field">
        <label>Seed</label>
        <input data-seed type="number" value="-1" />
      </div>
      <label class="toggle">
        <input data-enhance type="checkbox" />
        Enhance prompt locally
      </label>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-generate>Generate</button>
      <button type="button" class="mini-button" data-status>Status</button>
    </div>
    <div class="widget-note image-backend-note" data-image-guidance>
      Image generation uses Stable Diffusion. Llama vision is for understanding screenshots, not creating images.
    </div>
    <div class="image-preview"><img data-image alt="" /></div>
    <pre class="json-output compact" data-output></pre>
  `;
  body.querySelector("[data-prompt]").value = prompt;
  const generate = () => generateImageFromWidget(body);
  body.querySelector("[data-generate]").onclick = generate;
  body.querySelector("[data-status]").onclick = () => imageStatus(body);
  addMessage("assistant", "Image Lab is open.");
  addActivity("Image Lab opened", prompt.slice(0, 80));
  if (autoRun) await generate();
}

async function imageStatus(body) {
  const output = body.querySelector("[data-output]");
  output.textContent = "Checking image backend...";
  const modelPath = body.querySelector("[data-model-path]")?.value.trim() || "";
  const data = await api(`/api/imagegen/status${modelPath ? `?model_path=${encodeURIComponent(modelPath)}` : ""}`);
  renderImageBackendGuidance(body, data);
  output.textContent = jsonText(data);
  return data;
}

function renderImageBackendGuidance(body, data) {
  const root = body.querySelector("[data-image-guidance]");
  if (!root) return;
  const steps = data.next_steps?.length
    ? `<ul>${data.next_steps.map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ul>`
    : "";
  root.innerHTML = `
    <div class="button-row">
      <span class="badge ${data.ready ? "good" : "warn"}">${data.ready ? "Image backend ready" : "Image backend missing"}</span>
      <span class="badge ${data.automatic1111?.ready ? "good" : "warn"}">A1111 ${data.automatic1111?.ready ? "ready" : "off"}</span>
      <span class="badge ${data.diffusers?.ready && data.model_path?.exists ? "good" : "warn"}">Diffusers ${
        data.diffusers?.ready ? (data.model_path?.exists ? "ready" : "needs model") : "missing packages"
      }</span>
    </div>
    <span>${escapeHtml(data.note || "Image generation needs a local Stable Diffusion backend.")}</span>
    ${steps}
  `;
}

async function generateImageFromWidget(body) {
  const output = body.querySelector("[data-output]");
  const image = body.querySelector("[data-image]");
  image.classList.remove("ready");
  image.removeAttribute("src");
  output.textContent = "Generating with local backend...";
  const payload = {
    prompt: body.querySelector("[data-prompt]").value,
    negative_prompt: body.querySelector("[data-negative]").value,
    backend: body.querySelector("[data-backend]").value,
    width: Number(body.querySelector("[data-width]").value),
    height: Number(body.querySelector("[data-height]").value),
    steps: Number(body.querySelector("[data-steps]").value),
    guidance_scale: Number(body.querySelector("[data-cfg]").value),
    seed: Number(body.querySelector("[data-seed]").value),
    model_path: body.querySelector("[data-model-path]").value || null,
    enhance_prompt: body.querySelector("[data-enhance]").checked,
  };
  try {
    const status = await imageStatus(body);
    if (!status.ready) {
      output.textContent = jsonText(status);
      addMessage("assistant", "Image generation needs a Stable Diffusion backend or local model folder. Llama vision will not generate images.");
      addActivity("Image backend missing", "Stable Diffusion not configured");
      return;
    }
    output.textContent = "Generating with local backend...";
    const data = await api("/api/imagegen/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    output.textContent = jsonText(data);
    if (data.image_path) {
      image.src = `/api/artifact?path=${encodeURIComponent(data.image_path)}`;
      image.classList.add("ready");
    }
    addMessage("assistant", data.success ? "The image is ready." : data.error || "Image generation did not complete.");
    addActivity("Image generated", data.backend || payload.backend);
  } catch (err) {
    output.textContent = err.message;
    addMessage("assistant", `The image backend is not ready yet: ${err.message}`);
    addActivity("Image failed", "Backend unavailable");
  }
}

async function runInvestigation(subject, roots = []) {
  upsertContext("subject", "Subject", subject);
  if (roots[0]) upsertContext("root", "Root", roots[0]);
  const { body } = createWidget("Investigation", "Dossier");
  renderToolRun(body, {
    title: "Investigation run",
    status: "Running",
    detail: subject,
    steps: [
      { label: "Search cited evidence", state: "running" },
      { label: "Extract entities and timeline", state: "" },
      { label: "Apply hallucination guard", state: "" },
    ],
  });
  const data = await api("/api/investigate", {
    method: "POST",
    body: JSON.stringify({
      subject,
      roots,
      top_k: 8,
      max_file_results: 40,
    }),
  });
  body.innerHTML = renderInvestigation(data, subject, roots);
  body.querySelector("[data-report]").onclick = () => runReport(subject, roots);
  addMessage("assistant", `Dossier opened for ${subject}. Confidence: ${Math.round((data.confidence || 0) * 100)}%.`);
  addActivity("Investigation", subject);
}

function renderInvestigation(data, subject, roots) {
  const entities = data.entities || {};
  const aliases = entities.aliases?.join(", ") || "none";
  const contacts = [...(entities.emails || []), ...(entities.phones || []), ...(entities.handles || [])].join(", ") || "none";
  const flags = data.risk_flags?.join(", ") || "none";
  const actions = data.next_actions?.join("\n") || "none";
  return `
    <div class="button-row">
      <span class="badge">${Math.round((data.confidence || 0) * 100)}% confidence</span>
      <span class="badge ${data.hallucination_guard?.status === "supported" ? "good" : "warn"}">${escapeHtml(data.hallucination_guard?.status || "guarded")}</span>
      <button type="button" class="mini-button" data-report>Report</button>
    </div>
    <div class="cards">
      <div class="cardlet"><strong>Subject</strong><span>${escapeHtml(subject)}</span></div>
      <div class="cardlet"><strong>Summary</strong><span>${escapeHtml(data.summary || "")}</span></div>
      <div class="cardlet"><strong>Aliases</strong><span>${escapeHtml(aliases)}</span></div>
      <div class="cardlet"><strong>Contacts / Handles</strong><span>${escapeHtml(contacts)}</span></div>
      <div class="cardlet"><strong>Risk Flags</strong><span>${escapeHtml(flags)}</span></div>
      <div class="cardlet"><strong>Next Actions</strong><pre class="code-output compact">${escapeHtml(actions)}</pre></div>
    </div>
    <div>
      <span class="mini-label">Evidence</span>
      ${renderInvestigationEvidence(data.evidence || [])}
    </div>
    <details>
      <summary>Raw dossier</summary>
      ${renderOutput({ ...data, roots }, true)}
    </details>
  `;
}

function renderInvestigationEvidence(items) {
  if (!items.length) {
    return `<div class="result-item"><strong>No cited evidence</strong><span class="muted">Index more files or add folder roots.</span></div>`;
  }
  return `
    <div class="result-list">
      ${items
        .map(
          (item) => `
            <div class="result-item">
              <strong>${escapeHtml(item.file_name)} · ${escapeHtml(item.location)}</strong>
              <span class="muted">${escapeHtml(item.citation)} · score ${Number(item.score || 0).toFixed(3)}</span>
              <span>${escapeHtml(item.snippet || "")}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

async function runReport(subject, roots = []) {
  upsertContext("subject", "Subject", subject);
  if (roots[0]) upsertContext("root", "Root", roots[0]);
  const { body } = createWidget("Report Export", "Report");
  renderToolRun(body, {
    title: "Report export",
    status: "Running",
    detail: subject,
    steps: [
      { label: "Build dossier", state: "running" },
      { label: "Write Markdown, HTML, JSON", state: "" },
      { label: "Prepare artifact links", state: "" },
    ],
  });
  const data = await api("/api/report/generate", {
    method: "POST",
    body: JSON.stringify({ subject, roots, top_k: 8 }),
  });
  body.innerHTML = `
    <div class="result-list">
      <div class="result-item">
        <strong>HTML Report</strong>
        <a href="/api/artifact?path=${encodeURIComponent(data.html_path)}" target="_blank">Open report</a>
        <span class="muted">${escapeHtml(data.html_path)}</span>
      </div>
      <div class="result-item">
        <strong>Markdown Report</strong>
        <a href="/api/artifact?path=${encodeURIComponent(data.markdown_path)}" target="_blank">Open markdown</a>
        <span class="muted">${escapeHtml(data.markdown_path)}</span>
      </div>
      <div class="result-item">
        <strong>Raw JSON</strong>
        <a href="/api/artifact?path=${encodeURIComponent(data.json_path)}" target="_blank">Open JSON</a>
        <span class="muted">${escapeHtml(data.json_path)}</span>
      </div>
    </div>
    ${renderOutput(data, true)}
  `;
  addMessage("assistant", `Report exported for ${subject}.`);
  addActivity("Report exported", subject);
}

async function runAnalytics(query) {
  if (query) upsertContext("subject", "Subject", query);
  const { body } = createWidget("Evidence Analytics", "Analytics");
  renderToolRun(body, {
    title: "Analytics run",
    status: "Running",
    detail: query,
    steps: [
      { label: "Collect evidence chunks", state: "running" },
      { label: "Build graph and timeline", state: "" },
      { label: "Score signals", state: "" },
    ],
  });
  const data = await api(`/api/analytics?query=${encodeURIComponent(query)}&limit=80`);
  body.innerHTML = `
    <div class="button-row">
      <span class="badge">${Number(data.metrics?.chunks_analyzed || 0)} chunks</span>
      <span class="badge">${Number(data.metrics?.graph_nodes || 0)} nodes</span>
      <span class="badge">${Number(data.metrics?.timeline_items || 0)} timeline</span>
    </div>
    <div class="graph-wrap">
      <canvas data-graph width="900" height="360"></canvas>
    </div>
    <div class="split-two">
      <div>
        <span class="mini-label">Prediction Brief</span>
        <div data-prediction class="cards"></div>
      </div>
      <div>
        <span class="mini-label">Timeline</span>
        <div data-timeline class="timeline"></div>
      </div>
    </div>
    <details>
      <summary>Raw analytics</summary>
      ${renderOutput(data, true)}
    </details>
  `;
  renderPrediction(body.querySelector("[data-prediction]"), data.prediction || {}, data.metrics || {});
  renderTimeline(body.querySelector("[data-timeline]"), data.timeline || []);
  renderGraph(body.querySelector("[data-graph]"), data.graph || {});
  addMessage("assistant", "Analytics are open with graph, forecast, and timeline.");
  addActivity("Analytics", query);
}

function renderPrediction(root, prediction, metrics) {
  const signals = prediction.signals?.map((item) => `${item.signal} (${item.count})`).join(", ") || "No strong signals";
  const gaps = prediction.gaps?.length ? prediction.gaps.join("\n") : "No major evidence gaps detected.";
  const actions = prediction.next_actions?.join("\n") || "No actions available.";
  root.innerHTML = `
    <div class="cardlet"><strong>Confidence</strong><div class="score">${Math.round((prediction.confidence || 0) * 100)}%</div><span class="muted">${escapeHtml(prediction.caveat || "")}</span></div>
    <div class="cardlet"><strong>Forecast</strong><span>${escapeHtml(prediction.forecast || "")}</span></div>
    <div class="cardlet"><strong>Trend</strong><span>${escapeHtml(prediction.trend?.summary || "")}</span></div>
    <div class="cardlet"><strong>Signals</strong><span>${escapeHtml(signals)}</span></div>
    <div class="cardlet"><strong>Evidence Gaps</strong><pre class="code-output compact">${escapeHtml(gaps)}</pre></div>
    <div class="cardlet"><strong>Next Actions</strong><pre class="code-output compact">${escapeHtml(actions)}</pre></div>
    <div class="cardlet"><strong>Coverage</strong><span>${Number(metrics.chunks_analyzed || 0)} chunks · ${Number(metrics.timeline_items || 0)} timeline items</span></div>
  `;
}

function renderTimeline(root, items) {
  if (!items.length) {
    root.innerHTML = `<div class="timeline-item"><strong>No timeline items</strong><span class="muted">Index dated evidence or transcripts.</span></div>`;
    return;
  }
  root.innerHTML = items
    .slice(0, 40)
    .map(
      (item) => `
        <div class="timeline-item">
          <strong>${escapeHtml(item.when)} · ${escapeHtml(item.kind)}</strong>
          <span class="muted">${escapeHtml(item.source)} · ${escapeHtml(item.modality)} · ${(Number(item.confidence || 0) * 100).toFixed(0)}%</span>
          <span>${escapeHtml(item.summary)}</span>
          <span class="muted">${escapeHtml(item.citation)}</span>
        </div>
      `,
    )
    .join("");
}

function renderGraph(canvas, graph) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const theme = getComputedStyle(document.documentElement);
  const surface = theme.getPropertyValue("--surface-2").trim() || "#202020";
  const ink = theme.getPropertyValue("--ink-soft").trim() || "#ded8d4";
  const muted = theme.getPropertyValue("--muted").trim() || "#a19794";
  canvas.width = Math.max(640, Math.floor(rect.width * devicePixelRatio));
  canvas.height = Math.floor(360 * devicePixelRatio);
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const width = canvas.width / devicePixelRatio;
  const height = canvas.height / devicePixelRatio;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = surface;
  ctx.fillRect(0, 0, width, height);
  const nodes = (graph.nodes || []).slice(0, 36);
  const edges = (graph.edges || [])
    .filter((edge) => nodes.some((node) => node.id === edge.source) && nodes.some((node) => node.id === edge.target))
    .slice(0, 80);
  if (!nodes.length) {
    ctx.fillStyle = muted;
    ctx.font = "13px Segoe UI";
    ctx.fillText("No graph data yet.", 22, 34);
    return;
  }
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) * 0.38;
  const positions = new Map();
  nodes.forEach((node, index) => {
    const angle = (Math.PI * 2 * index) / nodes.length - Math.PI / 2;
    const ring = node.type === "document" ? radius * 0.45 : node.type === "chunk" ? radius * 0.72 : radius;
    positions.set(node.id, {
      x: centerX + Math.cos(angle) * ring,
      y: centerY + Math.sin(angle) * ring,
    });
  });
  ctx.lineWidth = 1;
  edges.forEach((edge) => {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return;
    ctx.strokeStyle = edge.type?.startsWith("cross") ? "rgba(255, 184, 107, .38)" : "rgba(156, 207, 255, .24)";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  nodes.forEach((node) => {
    const point = positions.get(node.id);
    const size = node.type === "document" ? 15 : node.type === "chunk" ? 11 : 8;
    ctx.fillStyle = colorForType(node.type);
    ctx.beginPath();
    ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = ink;
    ctx.font = "12px Segoe UI";
    ctx.fillText(String(node.label).slice(0, 22), point.x + size + 4, point.y + 4);
  });
}

function colorForType(type) {
  return {
    document: "#76d49b",
    chunk: "#9ccfff",
    entity: "#cdbdff",
    date: "#ffb86b",
    modality: "#ff8a3d",
  }[type] || "#a19794";
}

async function openVisionWidget(options) {
  const { body } = createWidget("Screen Vision", "Vision");
  body.innerHTML = `
    <div class="form-grid">
      <label class="toggle">
        <input data-active type="checkbox" />
        Active window
      </label>
      <label class="toggle">
        <input data-ingest type="checkbox" checked />
        Ingest result
      </label>
      <label class="toggle">
        <input data-vlm type="checkbox" checked />
        Use local VLM
      </label>
      <div class="field">
        <label>Vision model</label>
        <input data-model value="llama3.2-vision" />
      </div>
      <div class="field span-all">
        <label>Image path</label>
        <input data-image-path placeholder="C:\\path\\to\\image.png" />
      </div>
    </div>
    <div class="button-row">
      <button type="button" class="primary-button" data-capture>Capture</button>
      <button type="button" class="mini-button" data-analyze>Analyze path</button>
    </div>
    <div class="vision-preview"><img data-preview alt="" /></div>
    <pre class="json-output compact" data-output></pre>
  `;
  body.querySelector("[data-active]").checked = Boolean(options.activeWindow);
  body.querySelector("[data-image-path]").value = options.imagePath || "";
  body.querySelector("[data-capture]").onclick = () => captureVision(body);
  body.querySelector("[data-analyze]").onclick = () => analyzeVisionPath(body);
  addMessage("assistant", "Screen Vision is open.");
  addActivity("Vision opened", options.mode || "capture");
  if (options.auto && options.mode === "capture") await captureVision(body);
  if (options.auto && options.mode === "analyze") await analyzeVisionPath(body);
}

async function captureVision(body) {
  const output = body.querySelector("[data-output]");
  const preview = body.querySelector("[data-preview]");
  preview.classList.remove("ready");
  output.textContent = "Capturing screen...";
  const data = await api("/api/vision/screenshot", {
    method: "POST",
    body: JSON.stringify({
      active_window: body.querySelector("[data-active]").checked,
      analyze: true,
      ingest: body.querySelector("[data-ingest]").checked,
      use_vlm: body.querySelector("[data-vlm]").checked,
      vision_model: body.querySelector("[data-model]").value || null,
    }),
  });
  output.textContent = jsonText(data);
  if (data.screenshot?.image_path) {
    preview.src = `/api/artifact?path=${encodeURIComponent(data.screenshot.image_path)}`;
    preview.classList.add("ready");
  }
  if (body.querySelector("[data-ingest]").checked) await refreshSources();
  addMessage("assistant", "Screenshot captured and analyzed.");
  addActivity("Screenshot captured", data.screenshot?.image_path || "");
}

async function analyzeVisionPath(body) {
  const output = body.querySelector("[data-output]");
  const imagePath = body.querySelector("[data-image-path]").value.trim();
  if (!imagePath) {
    output.textContent = "No image path supplied.";
    return;
  }
  output.textContent = "Analyzing image...";
  const data = await api("/api/vision/analyze", {
    method: "POST",
    body: JSON.stringify({
      image: imagePath,
      ingest: body.querySelector("[data-ingest]").checked,
      use_vlm: body.querySelector("[data-vlm]").checked,
      vision_model: body.querySelector("[data-model]").value || null,
    }),
  });
  output.textContent = jsonText(data);
  if (body.querySelector("[data-ingest]").checked) await refreshSources();
  addMessage("assistant", "Image analysis is ready.");
  addActivity("Image analyzed", imagePath);
}

async function runWindows() {
  const { body } = createWidget("Visible Windows", "Operator");
  body.innerHTML = renderOutput("Reading visible windows...", true);
  const data = await api("/api/operator/windows");
  body.innerHTML = renderOperatorList(data.windows || [], "title", "process");
  addMessage("assistant", `${(data.windows || []).length} visible window${(data.windows || []).length === 1 ? "" : "s"} found.`);
  addActivity("Windows listed", `${(data.windows || []).length} found`);
}

async function runTabs() {
  const { body } = createWidget("Browser Tabs", "Operator");
  body.innerHTML = renderOutput("Checking browser tabs...", true);
  const data = await api("/api/operator/tabs");
  body.innerHTML = `
    ${renderOperatorList(data.tabs || [], "title", "url")}
    ${renderOutput(data.note || "", true)}
  `;
  addMessage("assistant", `${(data.tabs || []).length} browser tab${(data.tabs || []).length === 1 ? "" : "s"} found.`);
  addActivity("Tabs checked", `${(data.tabs || []).length} found`);
}

function renderOperatorList(items, titleKey, detailKey) {
  if (!items.length) {
    return `<div class="result-item"><strong>No items found</strong><span class="muted">Nothing visible from this local check.</span></div>`;
  }
  return `
    <div class="result-list">
      ${items
        .map(
          (item) => `
            <div class="result-item">
              <strong>${escapeHtml(item[titleKey] || item.name || item.path || "Item")}</strong>
              <span class="muted">${escapeHtml(item[detailKey] || item.path || "")}</span>
              ${item.path ? `<span class="muted">${escapeHtml(item.path)}</span>` : ""}
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

async function runScan(path) {
  upsertContext("root", "Root", path);
  const { body } = createWidget("Folder Scan", "Operator");
  body.innerHTML = renderOutput(`Scanning ${path}...`, true);
  const data = await api("/api/operator/scan", {
    method: "POST",
    body: JSON.stringify({ path, max_depth: 2, max_items: 100 }),
  });
  body.innerHTML = renderOperatorList(data.items || [], "name", "extension");
  addMessage("assistant", `Folder scan opened for ${path}.`);
  addActivity("Folder scanned", path);
}

async function runFind(path, query) {
  upsertContext("root", "Root", path);
  const { body } = createWidget("File Search", "Operator");
  body.innerHTML = renderOutput(`Searching ${path} for ${query}...`, true);
  const data = await api("/api/operator/find", {
    method: "POST",
    body: JSON.stringify({ path, query, content: true, max_depth: 5, max_results: 60 }),
  });
  body.innerHTML = renderOperatorList(data.matches || [], "name", "extension");
  addMessage("assistant", `${(data.matches || []).length} file match${(data.matches || []).length === 1 ? "" : "es"} found.`);
  addActivity("Files searched", query);
}

async function runOpenPath(path) {
  const { body } = createWidget("Open Path", "Operator");
  body.innerHTML = `
    <div class="button-row">
      <span class="badge warn">Dry run</span>
      <button type="button" class="mini-button" data-open>Open</button>
    </div>
    <pre class="json-output compact" data-output>Checking ${escapeHtml(path)}...</pre>
  `;
  const output = body.querySelector("[data-output]");
  const perform = async (execute) => {
    const data = await api("/api/operator/open", {
      method: "POST",
      body: JSON.stringify({ path, execute }),
    });
    output.textContent = jsonText(data);
    return data;
  };
  body.querySelector("[data-open]").onclick = () => {
    addApprovalCard({
      title: "Open local path?",
      subtitle: "Axiom will ask the operating system to open this path.",
      target: path,
      confirmLabel: "Open once",
      previewLabel: "Dry run",
      onPreview: () => perform(false),
      onConfirm: () => perform(true),
    });
  };
  await perform(false);
  addMessage("assistant", `Open request prepared for ${path}.`);
  addActivity("Open path", path);
}

async function openCommandWidget(command, execute) {
  const { body } = createWidget("Command", "Operator");
  body.innerHTML = `
    <div class="field">
      <label>Command</label>
      <input data-command />
    </div>
    <label class="toggle">
      <input data-execute type="checkbox" />
      Execute
    </label>
    <div class="button-row">
      <button type="button" class="primary-button" data-run>Run</button>
    </div>
    <pre class="json-output" data-output></pre>
  `;
  body.querySelector("[data-command]").value = command;
  body.querySelector("[data-execute]").checked = false;
  body.querySelector("[data-run]").onclick = () => runCommandFromWidget(body);
  await runCommandFromWidget(body);
  if (execute) {
    addApprovalCard({
      title: "Run local command?",
      subtitle: "Axiom will execute only if the command passes local policy.",
      target: command,
      confirmLabel: "Run once",
      previewLabel: "Policy check",
      onPreview: () => runCommandPayload(command, false),
      onConfirm: async () => {
        body.querySelector("[data-execute]").checked = true;
        return runCommandFromWidget(body);
      },
    });
  }
}

async function runCommandFromWidget(body) {
  const output = body.querySelector("[data-output]");
  const command = body.querySelector("[data-command]").value;
  output.textContent = "Running command policy check...";
  const data = await runCommandPayload(command, body.querySelector("[data-execute]").checked);
  output.textContent = jsonText(data);
  addMessage("assistant", data.executed ? "Command finished." : data.allowed ? "Command is ready to run." : "Command was blocked by policy.");
  addActivity("Command", data.executed ? "Executed" : "Checked");
  return data;
}

async function runCommandPayload(command, execute) {
  return api("/api/operator/run", {
    method: "POST",
    body: JSON.stringify({
      command: commandParts(command),
      execute,
      unsafe: false,
      shell: false,
    }),
  });
}

async function runPrompt(prompt) {
  const input = $("chatInput");
  input.value = prompt;
  resizeComposer();
  $("chatForm").requestSubmit();
}

async function submitChat(event) {
  event.preventDefault();
  const input = $("chatInput");
  const value = input.value.trim();
  const attachments = state.pendingAttachments.slice();
  if (!value && !attachments.length) return;
  input.value = "";
  state.pendingAttachments = [];
  renderAttachmentTray();
  resizeComposer();
  hideSlashMenu();
  hideEvidenceMenu();
  hideAttachMenu();
  const userText = value || attachmentSummary(attachments);
  if (!activeChat()) createWorkspaceChat();
  addMessage("user", userText, { attachments });
  setBusy(true);
  try {
    if (attachments.length) {
      await handleAttachments(value, attachments);
    } else if (value.startsWith("/")) {
      await handleSlashCommand(value);
    } else {
      const intent = detectIntent(value);
      await handleIntent(intent, value);
    }
  } catch (err) {
    addMessage("assistant", err.message);
    addActivity("Error", err.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

function attachmentSummary(attachments) {
  if (!attachments.length) return "";
  const folderCount = attachments.filter((attachment) => attachment.folder).length;
  const fileCount = attachments.filter((attachment) => !attachment.folder && attachment.source !== "clipboard").length;
  const snipCount = attachments.filter((attachment) => attachment.source === "clipboard").length;
  if (folderCount) return `${attachments.length} folder item${attachments.length === 1 ? "" : "s"}`;
  if (fileCount) return `${attachments.length} selected file${attachments.length === 1 ? "" : "s"}`;
  return snipCount === 1 ? "Pasted screenshot" : `${snipCount} pasted screenshots`;
}

function resizeComposer() {
  const input = $("chatInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function wire() {
  $("chatForm").addEventListener("submit", submitChat);
  $("attachButton").onclick = () => {
    const menu = ensureAttachMenu();
    if (menu.hidden) showAttachMenu();
    else hideAttachMenu();
  };
  $("fileInput").addEventListener("change", async (event) => {
    const files = [...event.target.files];
    if (files.length) await addPendingFiles(files, { source: "file" });
    event.target.value = "";
    $("chatInput").focus();
  });
  $("folderInput").addEventListener("change", async (event) => {
    const files = [...event.target.files];
    if (files.length) await addPendingFiles(files, { source: "folder", folder: true });
    event.target.value = "";
    $("chatInput").focus();
  });
  $("workspaceFolderInput").addEventListener("change", (event) => {
    const files = [...event.target.files];
    handleWorkspaceFolderFiles(files);
    event.target.value = "";
    $("chatInput").focus();
  });
  document.addEventListener("paste", async (event) => {
    const files = filesFromTransfer(event.clipboardData, { imagesOnly: true });
    if (!files.length) return;
    event.preventDefault();
    await addPendingFiles(files, { source: "clipboard" });
    $("chatInput").focus();
  });
  $("chatForm").addEventListener("dragover", (event) => {
    const types = [...(event.dataTransfer?.types || [])];
    if (types.includes("application/x-axiom-source") || filesFromTransfer(event.dataTransfer).length) {
      event.preventDefault();
      $("chatForm").classList.add("drag-ready");
      event.dataTransfer.dropEffect = "copy";
    }
  });
  $("chatForm").addEventListener("dragleave", () => {
    $("chatForm").classList.remove("drag-ready");
  });
  $("chatForm").addEventListener("drop", async (event) => {
    const reference = event.dataTransfer.getData("application/x-axiom-source");
    if (reference) {
      event.preventDefault();
      $("chatForm").classList.remove("drag-ready");
      insertComposerText(`${reference} `);
      $("chatInput").focus();
      return;
    }
    const files = filesFromTransfer(event.dataTransfer);
    if (!files.length) return;
    event.preventDefault();
    $("chatForm").classList.remove("drag-ready");
    await addPendingFiles(files, { source: "file" });
    $("chatInput").focus();
  });
  $("chatInput").addEventListener("input", () => {
    resizeComposer();
    updateComposerMenus();
  });
  $("chatInput").addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideSlashMenu();
      hideEvidenceMenu();
      return;
    }
    const evidenceMenu = document.querySelector(".evidence-menu");
    const firstEvidenceOption = evidenceMenu && !evidenceMenu.hidden ? evidenceMenu.querySelector("[data-evidence-index]") : null;
    if (event.key === "Enter" && firstEvidenceOption) {
      event.preventDefault();
      firstEvidenceOption.click();
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chatForm").requestSubmit();
    }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".slash-menu") && event.target !== $("chatInput")) hideSlashMenu();
    if (!event.target.closest(".evidence-menu") && event.target !== $("chatInput")) hideEvidenceMenu();
    if (!event.target.closest(".attach-menu") && event.target !== $("attachButton")) hideAttachMenu();
  });
  $("newChatButton").onclick = () => createWorkspaceChat();
  $("addFolderButton").onclick = () => $("workspaceFolderInput").click();
  $("evidenceRailButton").onclick = () => toggleEvidenceDrawer();
  $("refreshSources").onclick = async () => {
    const sources = await refreshSources();
    addActivity("Evidence refreshed", `${sources.length} indexed`);
  };
  $("sourcesButton").onclick = openSourcesWidget;
  $("setupButton").onclick = openSetupWidget;
  $("questionButton").onclick = openQuestionBuilder;
  $("contextButton").onclick = openContextWidget;
  $("clearWidgets").onclick = clearWidgets;
  $("sidePanelToggle").onclick = toggleSidePanel;
}

loadWorkspace();
wire();
renderActivity();
renderContextChips();
renderQuickActions();
renderFolderTree();
renderActiveChat();
refreshSetup().catch((err) => addMessage("assistant", err.message));
refreshSources().catch(() => {});
