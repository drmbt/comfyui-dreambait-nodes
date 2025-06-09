import { app } from "../../scripts/app.js"
import { ComfyWidgets } from "../../scripts/widgets.js"

app.registerExtension({
  name: "Comfy.MarkdownRenderer",
  async nodeCreated(node) {
    try {
      if (node && node.comfyClass === "MarkdownRenderer") {
        // Add emoji to node title
        node.title = "📝 " + (node.title || "Markdown Renderer")

        // Initialize editing state
        node._hasInputConnection = false
        node._editableContent = ""
        node._hasReceivedData = false

        // Monitor for input connections
        const originalOnConnectionsChange = node.onConnectionsChange
        node.onConnectionsChange = function (type, index, connected, link_info) {
          if (originalOnConnectionsChange) {
            originalOnConnectionsChange.call(this, type, index, connected, link_info)
          }

          // Check if input is connected
          if (type === 1 && index === 0) { // type 1 = input, index 0 = first input
            this._hasInputConnection = connected
            console.log("[MarkdownRenderer] Input connection changed:", connected)

            if (!connected) {
              // Input disconnected - show editor with current content
              this._hasReceivedData = false
              showEditor.call(this)
            } else {
              // Input connected - reset data received state and show waiting overlay
              this._hasReceivedData = false
              console.log("[MarkdownRenderer] Input connected - will show backend rendered content")
              showWaitingForInput.call(this)
            }
          }
        }

        // Initialize with editor if no input connection
        console.log("[MarkdownRenderer] Initializing node, hasInputConnection:", node._hasInputConnection)
        if (!node._hasInputConnection) {
          console.log("[MarkdownRenderer] Calling showEditor")
          setTimeout(() => {
            console.log("[MarkdownRenderer] showEditor timeout executing")
            showEditor.call(node)
          }, 100)
        } else {
          console.log("[MarkdownRenderer] Input connection detected, showing waiting overlay")
          setTimeout(() => {
            showWaitingForInput.call(node)
          }, 100)
        }
      }
    } catch (error) {
      console.error("[MarkdownRenderer] Error in nodeCreated:", error)
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "MarkdownRenderer") {
      // Add custom CSS for markdown styling
      const style = document.createElement('style')
      style.textContent = `
        .markdown-content {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
          line-height: 1.6;
          color: #e0e0e0;
          padding: 16px 12px 12px 12px;
          width: 100%;
          height: 100%;
          background: #0d0d0d;
          border-radius: 6px;
          overflow-y: auto;
          overflow-x: hidden;
          box-sizing: border-box;
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
        }
        .markdown-content::-webkit-scrollbar {
          width: 6px;
        }
        .markdown-content::-webkit-scrollbar-track {
          background: transparent;
        }
        .markdown-content::-webkit-scrollbar-thumb {
          background: rgba(255,255,255,0.2);
          border-radius: 3px;
        }
        .markdown-content::-webkit-scrollbar-thumb:hover {
          background: rgba(255,255,255,0.3);
        }
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
          margin-top: 1.2em;
          margin-bottom: 0.6em;
          color: #ffffff;
          font-weight: 600;
        }
        .markdown-content h1 { color: #00d4ff; }
        .markdown-content h2 { color: #00b8e6; }
        .markdown-content h3 { color: #009fcc; }
        .markdown-content code {
          background: #1a1a1a;
          padding: 2px 4px;
          border-radius: 3px;
          font-family: 'SF Mono', Monaco, Consolas, monospace;
          color: #00d4ff;
          border: 1px solid #333;
        }
        .markdown-content pre {
          background: #0a0a0a;
          padding: 12px;
          border-radius: 4px;
          overflow-x: auto;
          color: #e0e0e0;
          border: 1px solid #1a1a1a;
        }
        .markdown-content blockquote {
          border-left: 2px solid #007acc;
          margin: 0;
          padding-left: 12px;
          color: #bbb;
          background: #0a0a0a;
          padding: 8px 12px;
          border-radius: 0 3px 3px 0;
        }
        .markdown-content table {
          border-collapse: collapse;
          width: 100%;
          margin: 1em 0;
          border-radius: 4px;
          overflow: hidden;
          border: 1px solid #222;
        }
        .markdown-content th, .markdown-content td {
          border: 1px solid #222;
          padding: 8px;
        }
        .markdown-content th {
          background: #111;
          color: #00d4ff;
          font-weight: 600;
        }
        .markdown-content a {
          color: #007acc;
          text-decoration: none;
          transition: color 0.2s ease;
        }
        .markdown-content a:hover {
          color: #00b8e6;
        }
                 .markdown-editor {
           width: 100%;
           height: 100%;
           background: #0d0d0d;
           border: 1px solid #222;
           border-radius: 6px;
           display: flex;
           flex-direction: column;
           box-sizing: border-box;
           position: absolute;
           top: 0;
           left: 0;
           right: 0;
           bottom: 0;
         }
         .markdown-editor-toolbar {
           background: #141414;
           color: #ffffff;
           padding: 6px 6px;
           font-size: 11px;
           display: flex;
           justify-content: space-between;
           align-items: center;
           border-bottom: 1px solid #1a1a1a;
           border-radius: 6px 6px 0 0;
           flex-shrink: 0;
           z-index: 100;
           position: relative;
           height: 32px;
           box-sizing: border-box;
         }
         .markdown-char-count {
           color: #666;
           font-size: 10px;
           font-family: 'SF Mono', Monaco, Consolas, monospace;
           background: #111;
           padding: 3px 4px;
           border-radius: 3px;
           border: 1px solid #222;
         }
         .markdown-editor-textarea {
           flex: 1;
           background: #0d0d0d;
           color: #e0e0e0;
           border: none;
           border-radius: 6px;
           resize: none;
           padding: 12px;
           font-family: 'SF Mono', Monaco, Consolas, monospace;
           font-size: 13px;
           outline: none;
           overflow-y: auto;
           line-height: 1.4;
         }
                  .markdown-toggle-group {
           display: flex;
           border-radius: 4px;
           overflow: hidden;
           border: 1px solid #222;
           background: #111;
         }
         .markdown-editor-button {
           background: transparent;
           color: #888;
           border: none;
           padding: 4px 6px;
           cursor: pointer;
           font-size: 10px;
           font-weight: 500;
           transition: all 0.2s ease;
           border-right: 1px solid #222;
           border-radius: 0;
           position: relative;
           text-transform: uppercase;
           letter-spacing: 0.5px;
           height: 20px;
           display: flex;
           align-items: center;
           justify-content: center;
         }
         .markdown-editor-button:last-child {
           border-right: none;
         }
         .markdown-editor-button:hover:not(.active) {
           background: #1a1a1a;
           color: #ccc;
         }
         .markdown-editor-button.active {
           background: #007acc;
           color: #ffffff;
           box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
         }
          .markdown-waiting-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            text-align: center;
            z-index: 1000;
            backdrop-filter: blur(2px);
            border: 1px solid #333;
          }
          .markdown-waiting-icon {
            font-size: 42px;
            margin-bottom: 12px;
            animation: pulse 2s ease-in-out infinite;
            filter: grayscale(30%);
          }
          .markdown-waiting-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 6px;
            color: #00d4ff;
          }
          .markdown-waiting-subtitle {
            font-size: 13px;
            opacity: 0.8;
            margin-bottom: 16px;
            max-width: 280px;
            line-height: 1.3;
            color: #ccc;
          }
          .markdown-waiting-note {
            font-size: 11px;
            opacity: 0.8;
            background: #111;
            padding: 6px 12px;
            border-radius: 12px;
            border: 1px solid #333;
            color: #00d4ff;
          }
          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.7; }
          }
      `
      document.head.appendChild(style)

      // Shared markdown widget creator (moved to global scope)
      window.createMarkdownWidget = function (node, config) {
        const {
          widgetName = 'markdown_widget',
          isEditable = false,
          initialContent = '',
          htmlContent = '',
          sourceText = '',
          onContentChange = null
        } = config

        // Create main container
        const mainContainer = document.createElement('div')
        mainContainer.style.cssText = `
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          position: absolute;
          top: 0; left: 0; right: 0; bottom: 0;
          background: #0d0d0d;
          border: 1px solid #222;
          border-radius: 6px;
          box-sizing: border-box;
        `

        // Create toolbar
        const toolbar = document.createElement('div')
        toolbar.className = 'markdown-editor-toolbar'

        const charCount = document.createElement('div')
        charCount.className = 'markdown-char-count'

        const toggleGroup = document.createElement('div')
        toggleGroup.className = 'markdown-toggle-group'

        const markdownButton = document.createElement('button')
        markdownButton.className = 'markdown-editor-button active'
        markdownButton.textContent = 'MD'

        const textButton = document.createElement('button')
        textButton.className = 'markdown-editor-button'
        textButton.textContent = 'Text'

        // Create content containers
        const container = document.createElement('div')
        container.className = 'markdown-content'
        container.style.flex = '1'
        container.innerHTML = htmlContent || parseMarkdownSimple(initialContent)

        const textarea = document.createElement('textarea')
        textarea.className = 'markdown-editor-textarea'
        textarea.style.display = 'none'
        textarea.style.flex = '1'
        textarea.readOnly = !isEditable
        textarea.placeholder = isEditable ? 'Enter your markdown content here...' : 'Source markdown from connected input...'
        textarea.value = sourceText || initialContent

        let isSourceMode = false
        let currentContent = initialContent

        // Update character count
        const updateCharCount = () => {
          const text = currentContent || ''
          charCount.textContent = `${text.length} chars`
        }
        updateCharCount()

        // Mode switching functions
        function showMarkdown() {
          container.style.display = 'block'
          textarea.style.display = 'none'
          markdownButton.classList.add('active')
          textButton.classList.remove('active')
          isSourceMode = false
          if (isEditable) {
            container.innerHTML = parseMarkdownSimple(currentContent)
          }
          updateCharCount()
        }

        function showText() {
          textarea.style.display = 'block'
          container.style.display = 'none'
          textButton.classList.add('active')
          markdownButton.classList.remove('active')
          isSourceMode = true
          updateCharCount()
        }

        // Event handlers - always toggle to the other mode
        markdownButton.addEventListener('click', (e) => {
          e.stopPropagation()
          if (isSourceMode) {
            showMarkdown()
          } else {
            showText()
          }
        })

        textButton.addEventListener('click', (e) => {
          e.stopPropagation()
          if (isSourceMode) {
            showMarkdown()
          } else {
            showText()
          }
        })

        // Click-to-edit functionality for editable widgets
        if (isEditable) {
          container.addEventListener('click', (e) => {
            // Prevent triggering if click originated from toolbar
            if (toolbar.contains(e.target)) {
              return
            }
            // Only trigger when in markdown mode (not when already in text mode)
            if (!isSourceMode) {
              showText()
            }
          })
          container.style.cursor = 'text'

          // Prevent textarea clicks from bubbling up and triggering mode switches
          textarea.addEventListener('click', (e) => {
            e.stopPropagation()
          })
        }

        if (isEditable) {
          textarea.addEventListener('input', () => {
            currentContent = textarea.value
            updateCharCount()
            if (onContentChange) onContentChange(currentContent)
          })

          // Auto-save functionality
          const autoSaveAndPreview = () => {
            if (isSourceMode) {
              if (textarea.value !== currentContent) {
                currentContent = textarea.value
                if (onContentChange) onContentChange(currentContent)
              }
              showMarkdown()
            }
          }

          const originalOnDeselected = node.onDeselected
          node.onDeselected = function () {
            autoSaveAndPreview()
            if (originalOnDeselected) originalOnDeselected.call(this)
          }

          const handleDocumentClick = (e) => {
            if (!mainContainer.contains(e.target)) {
              autoSaveAndPreview()
            }
          }
          document.addEventListener('click', handleDocumentClick)
        }

        // Assemble UI
        toggleGroup.appendChild(markdownButton)
        toggleGroup.appendChild(textButton)
        toolbar.appendChild(charCount)
        toolbar.appendChild(toggleGroup)
        mainContainer.appendChild(toolbar)
        mainContainer.appendChild(container)
        mainContainer.appendChild(textarea)

        // Create widget config
        const widgetConfig = {
          getValue: () => currentContent,
          setValue: (value) => {
            currentContent = value || ''
            if (textarea) textarea.value = currentContent
            if (container && !htmlContent) {
              container.innerHTML = parseMarkdownSimple(currentContent)
            }
            updateCharCount()
          },
          getHeight: () => 350,
          hideOnZoom: false
        }

        if (isEditable) {
          widgetConfig.onRemove = () => {
            document.removeEventListener('click', handleDocumentClick)
          }
        }

        return node.addDOMWidget(widgetName, 'STRING', mainContainer, widgetConfig)
      }

      // Make parseMarkdownSimple available globally for the widget creator
      window.parseMarkdownSimple = parseMarkdownSimple

      function populate(html) {
        if (this.widgets) {
          for (let i = 0; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.()
          }
          this.widgets.length = 0
        }

        const v = [...html]
        if (!v[0]) v.shift()

        for (let list of v) {
          if (!(list instanceof Array)) list = [list]
          for (const l of list) {
            // Get source text for character count
            let sourceText = 'Source content not available from connected input.'
            if (this._sourceText && this._sourceText.length > 0) {
              const sourceIndex = this.widgets?.length || 0
              sourceText = this._sourceText[sourceIndex] || this._sourceText[0] || sourceText
            } else if (this.inputs && this.inputs[0] && this.inputs[0].link) {
              const link = app.graph.links[this.inputs[0].link]
              if (link) {
                const sourceNode = app.graph.getNodeById(link.origin_id)
                if (sourceNode && sourceNode.widgets) {
                  const sourceWidget = sourceNode.widgets.find(w => w.name === 'text' || w.value)
                  if (sourceWidget && sourceWidget.value) {
                    sourceText = sourceWidget.value
                  }
                }
              }
            }

            window.createMarkdownWidget(this, {
              widgetName: 'text_' + (this.widgets?.length ?? 0),
              isEditable: false,
              htmlContent: l,
              sourceText: sourceText,
              initialContent: sourceText
            })
          }
        }

        // Update node size
        requestAnimationFrame(() => {
          const sz = this.computeSize()
          if (sz[0] < this.size[0]) sz[0] = this.size[0]
          if (sz[1] < this.size[1]) sz[1] = this.size[1]
          this.onResize?.(sz)
          app.graph.setDirtyCanvas(true, false)
        })
      }

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments)
        console.log("[MarkdownRenderer] Node executed with message:", message)

        if (message.html) {
          // Store the original text for source viewing and mark as received
          this._sourceText = message.text || []
          this._hasReceivedData = true
          populate.call(this, message.html)
        }
      }

      const VALUES = Symbol()
      const configure = nodeType.prototype.configure
      nodeType.prototype.configure = function () {
        // Store unmodified widget values as they get removed on configure by new frontend
        this[VALUES] = arguments[0]?.widgets_values
        return configure?.apply(this, arguments)
      }

      const onConfigure = nodeType.prototype.onConfigure
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments)
        const widgets_values = this[VALUES]
        if (widgets_values?.length) {
          // In newer frontend there seems to be a delay in creating the initial widget
          requestAnimationFrame(() => {
            populate.call(this, widgets_values)
          })
        }
      }
    }
  }
})

function showEditor(node) {
  console.log("[MarkdownRenderer] showEditor called, this:", this)

  // Remove any existing widgets
  if (this.widgets) {
    console.log("[MarkdownRenderer] Removing existing widgets:", this.widgets.length)
    for (let i = 0; i < this.widgets.length; i++) {
      this.widgets[i].onRemove?.()
    }
    this.widgets.length = 0
  }

  // Set default content if none exists
  if (!this._editableContent) {
    this._editableContent = `
Write your **markdown** content here!

- Bullet points
- *Italic text*
- \`Code snippets\`
`
  }

  const widget = window.createMarkdownWidget(this, {
    widgetName: 'markdown_editor',
    isEditable: true,
    initialContent: this._editableContent,
    onContentChange: (content) => {
      this._editableContent = content
    }
  })

  console.log("[MarkdownRenderer] Editor widget created:", widget)

  // Set minimum node size
  if (this.size[0] < 400) this.size[0] = 400
  if (this.size[1] < 350) this.size[1] = 350
}

function showWaitingForInput() {
  console.log("[MarkdownRenderer] showWaitingForInput called")

  // Remove any existing widgets
  if (this.widgets) {
    console.log("[MarkdownRenderer] Removing existing widgets:", this.widgets.length)
    for (let i = 0; i < this.widgets.length; i++) {
      this.widgets[i].onRemove?.()
    }
    this.widgets.length = 0
  }

  // Create waiting overlay container
  const mainContainer = document.createElement('div')
  mainContainer.style.cssText = `
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: #0d0d0d;
    border: 1px solid #222;
    border-radius: 6px;
    box-sizing: border-box;
  `

  // Create the waiting overlay
  const overlay = document.createElement('div')
  overlay.className = 'markdown-waiting-overlay'

  const icon = document.createElement('div')
  icon.className = 'markdown-waiting-icon'
  icon.textContent = '⏳'

  const title = document.createElement('div')
  title.className = 'markdown-waiting-title'
  title.textContent = 'Waiting for Input'

  const subtitle = document.createElement('div')
  subtitle.className = 'markdown-waiting-subtitle'
  subtitle.textContent = 'This node is connected to an input source. Execute the workflow to see the rendered markdown content.'

  const note = document.createElement('div')
  note.className = 'markdown-waiting-note'
  note.textContent = 'Manual content will be overridden'

  overlay.appendChild(icon)
  overlay.appendChild(title)
  overlay.appendChild(subtitle)
  overlay.appendChild(note)
  mainContainer.appendChild(overlay)

  // Create widget config
  const widgetConfig = {
    getValue: () => '',
    setValue: () => { },
    getHeight: () => 350,
    hideOnZoom: false
  }

  const widget = this.addDOMWidget('waiting_overlay', 'STRING', mainContainer, widgetConfig)

  console.log("[MarkdownRenderer] Waiting overlay widget created:", widget)

  // Set minimum node size
  if (this.size[0] < 400) this.size[0] = 400
  if (this.size[1] < 350) this.size[1] = 350
}

function hideEditor() {
  // Remove editor widgets
  if (this.widgets) {
    for (let i = 0; i < this.widgets.length; i++) {
      if (this.widgets[i].name === 'markdown_editor') {
        this.widgets[i].onRemove?.()
        this.widgets.splice(i, 1)
        break
      }
    }
  }
}

// Simple markdown parser for local preview (before backend processing)
function parseMarkdownSimple(text) {
  if (!text) return ''

  // Escape HTML first
  text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  // Headers
  text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>')
  text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>')
  text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>')

  // Bold and italic
  text = text.replace(/\*\*\*(.*?)\*\*\*/g, '<strong><em>$1</em></strong>')
  text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  text = text.replace(/\*(.*?)\*/g, '<em>$1</em>')

  // Code blocks (fenced)
  text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')

  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>')

  // Blockquotes
  text = text.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>')

  // Lists
  text = text.replace(/^[\*\-] (.*$)/gim, '<li>$1</li>')
  text = text.replace(/^(\d+)\. (.*$)/gim, '<li>$2</li>')

  // Wrap consecutive list items
  text = text.replace(/(<li>.*<\/li>)/gs, (match) => {
    return '<ul>' + match + '</ul>'
  })

  // Simple table support (basic)
  text = text.replace(/\|(.+)\|/g, (match, content) => {
    const cells = content.split('|').map(cell => cell.trim())
    return '<tr>' + cells.map(cell => `<td>${cell}</td>`).join('') + '</tr>'
  })
  text = text.replace(/(<tr>.*<\/tr>)/gs, '<table border="1">$1</table>')

  // Line breaks and paragraphs
  text = text.replace(/\n\n/g, '</p><p>')
  text = text.replace(/\n/g, '<br>')
  text = '<p>' + text + '</p>'

  // Clean up empty paragraphs
  text = text.replace(/<p><\/p>/g, '')
  text = text.replace(/<p>(<h[1-6]>)/g, '$1')
  text = text.replace(/(<\/h[1-6]>)<\/p>/g, '$1')
  text = text.replace(/<p>(<blockquote>)/g, '$1')
  text = text.replace(/(<\/blockquote>)<\/p>/g, '$1')
  text = text.replace(/<p>(<pre>)/g, '$1')
  text = text.replace(/(<\/pre>)<\/p>/g, '$1')
  text = text.replace(/<p>(<ul>)/g, '$1')
  text = text.replace(/(<\/ul>)<\/p>/g, '$1')
  text = text.replace(/<p>(<table>)/g, '$1')
  text = text.replace(/(<\/table>)<\/p>/g, '$1')

  return text
}  