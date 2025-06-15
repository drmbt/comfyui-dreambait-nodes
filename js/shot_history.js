import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

app.registerExtension({
    name: "Comfy.ShotHistory",
    async nodeCreated(node) {
        if (node.comfyClass === "ShotHistory") {
            addShotHistoryUI(node);
        }
    }
});

function addShotHistoryUI(node) {
    // Node state
    node.selectedFiles = []; // Changed to array for ordering
    node.fileList = [];
    node.filteredFileList = []; // Filtered list for search
    node.thumbnails = {};
    node.fileInfos = {}; // Cache for file information
    node.currentPath = "";
    node.scrollOffset = 0;
    node.thumbnailsLoaded = false;
    node.hoveredThumbnail = null;
    node.selectedThumbnail = null; // For info display
    node.isDraggingScrollbar = false;
    node.scrollbarWidth = 12;
    node.viewMode = "grid"; // "grid" or "list"
    node.searchFilter = "";
    node.showInfo = false;
    node.sortField = "filename"; // Current sort field
    node.sortDirection = "asc"; // "asc" or "desc"
    node.infoPanel = null; // DOM element for info panel
    
    // UI constants - will be dynamically calculated
    const THUMBNAIL_SIZE = 80;
    const THUMBNAIL_PADDING = 8;
    const GRID_PADDING = 10;
    let TOP_PADDING = 120; // Start higher to avoid button overlap
    const BOTTOM_PADDING = 20;
    
    // Colors
    const COLORS = {
        background: "#1e1e1e",
        thumbnail: "#2a2a2a",
        thumbnailHover: "#3a3a3a",
        thumbnailSelected: "#007acc",
        text: "#cccccc",
        border: "#555555"
    };

    // Get widgets
    const pathWidget = node.widgets.find(w => w.name === "path");
    const selectedFilesWidget = node.widgets.find(w => w.name === "selected_files");
    
    // Make selected_files widget hidden
    if (selectedFilesWidget) {
        selectedFilesWidget.hidden = true;
    }

    // Add search widget (keep this as a widget since it needs text input)
    const searchWidget = node.addWidget("text", "🔍 Search", "", (value) => {
        node.searchFilter = value.toLowerCase();
        applySearchFilter();
        node.setDirtyCanvas(true);
    });
    searchWidget.options = { placeholder: "Filter images..." };

    // Set initial size and make it properly resizable
    node.size = [450, 400];
    node.resizable = true;
    
    // Calculate dynamic values based on node size
    function calculateLayout() {
        const availableWidth = node.size[0] - (GRID_PADDING * 2) - node.scrollbarWidth;
        const thumbnailWidth = THUMBNAIL_SIZE + THUMBNAIL_PADDING;
        const thumbnailsPerRow = Math.max(1, Math.floor(availableWidth / thumbnailWidth));
        
        // Update TOP_PADDING based on number of widgets + toolbar
        const widgetCount = node.widgets ? node.widgets.length : 0;
        const toolbarHeight = 40; // Height for the toolbar
        TOP_PADDING = 80 + (widgetCount * 22) + toolbarHeight; // Dynamic padding based on widgets + toolbar
        
        
        // Additional space for info panel if visible
        if (node.showInfo) {
            TOP_PADDING += 120;
        }
        
        return { thumbnailsPerRow, availableWidth };
    }

    function drawToolbar(ctx) {
        const toolbarY = node.widgets ? 80 + (node.widgets.length * 22) : 80;
        const toolbarHeight = 40;
        
        // Draw toolbar background
        ctx.fillStyle = "rgba(45, 45, 45, 0.9)";
        ctx.fillRect(0, toolbarY, node.size[0], toolbarHeight);
        
        // Define toolbar icons
        const icons = [
            { icon: node.viewMode === "grid" ? '≡' : '⊞', tooltip: 'Toggle View Mode', fontSize: '16px' },
            { icon: '↻', tooltip: 'Refresh', fontSize: '16px' },
            { icon: '+', tooltip: 'Add Image', fontSize: '16px' },
            { icon: '✓', tooltip: 'Select All', fontSize: '14px' },
            { icon: '✗', tooltip: 'Clear All', fontSize: '14px' },
            { icon: node.showInfo ? '✕' : 'ℹ', tooltip: 'Toggle Info Panel', fontSize: '14px' }
        ];
        
        drawToolbarIcons(ctx, icons, toolbarY, toolbarHeight);
    }

    function drawToolbarIcons(ctx, icons, toolbarY, toolbarHeight) {
        const iconWidth = 32;
        const iconSpacing = 8;
        const rightMargin = 20;
        let xPosition = node.size[0] - iconWidth - rightMargin;

        icons.forEach((iconData, index) => {
            // Draw icon background on hover
            if (node.hoveredToolbarIcon === index) {
                ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
                ctx.fillRect(xPosition - 4, toolbarY + 4, iconWidth, toolbarHeight - 8);
            }
            
            ctx.fillStyle = '#ffffff';
            ctx.font = (iconData.fontSize || '16px') + ' Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(iconData.icon, xPosition + iconWidth / 2, toolbarY + toolbarHeight / 2);
            xPosition -= (iconWidth + iconSpacing);
        });
        
        // Store toolbar info for click detection
        node.toolbarIcons = icons;
        node.toolbarY = toolbarY;
        node.toolbarHeight = toolbarHeight;
        node.toolbarIconWidth = iconWidth;
        node.toolbarIconSpacing = iconSpacing;
        node.toolbarRightMargin = rightMargin;
    }

    function getClickedToolbarIcon(localX) {
        if (!node.toolbarIcons) return -1;
        
        const iconWidth = node.toolbarIconWidth;
        const iconSpacing = node.toolbarIconSpacing;
        const rightMargin = node.toolbarRightMargin;
        let xPosition = node.size[0] - iconWidth - rightMargin;

        for (let i = 0; i < node.toolbarIcons.length; i++) {
            if (localX >= xPosition && localX <= xPosition + iconWidth) {
                return i;
            }
            xPosition -= (iconWidth + iconSpacing);
        }
        
        return -1;
    }

    function handleToolbarClick(iconIndex) {
        const actions = [
            () => toggleViewMode(),        // Toggle View Mode
            () => {                        // Refresh
                console.log("Refresh button clicked");
                updateFileList(true);
            },
            () => openFileDialog(),        // Add Image
            () => selectAll(),             // Select All
            () => clearAll(),              // Clear All
            () => toggleInfoPanel()        // Toggle Info Panel
        ];
        
        if (actions[iconIndex]) {
            actions[iconIndex]();
            node.setDirtyCanvas(true);
        }
    }

    async function updateFileList(forceRefresh = false) {
        if (!pathWidget) return;
        
        const path = pathWidget.value || "./shots";
        
        if (!forceRefresh && path === node.currentPath && node.thumbnailsLoaded) {
            return; // No need to update unless forced
        }
        
        console.log(`Updating file list for path: ${path}`);
        
        // Reset state
        node.thumbnailsLoaded = false;
        node.fileList = [];
        node.thumbnails = {};
        node.fileInfos = {}; // Clear cached file info
        node.setDirtyCanvas(true);
        
        try {
            const response = await api.fetchApi("/shot_history/get_files", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: path })
            });
            
            const data = await response.json();
            if (data.files) {
                node.fileList = data.files;
                node.currentPath = path;
                console.log(`Found ${data.files.length} files`);
                applySearchFilter(); // Apply current search filter
                
                // Load all file info and thumbnails in parallel for better performance
                await Promise.all([
                    loadAllFileInfo(),
                    loadThumbnails()
                ]);
                
                node.setDirtyCanvas(true);
            }
        } catch (error) {
            console.error("Error updating file list:", error);
            node.fileList = [];
            node.thumbnailsLoaded = true;
            node.setDirtyCanvas(true);
        }
    }

    async function loadAllFileInfo() {
        if (!node.fileList || node.fileList.length === 0) return;
        
        console.log(`Loading file info for ${node.fileList.length} files...`);
        
        try {
            // Try to get all file info in a single batch request
            const response = await api.fetchApi("/shot_history/get_all_file_info", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ 
                    path: node.currentPath, 
                    filenames: node.fileList 
                })
            });
            
            if (response.ok) {
                const allInfo = await response.json();
                node.fileInfos = allInfo;
                console.log(`Loaded batch file info for ${Object.keys(allInfo).length} files`);
                return;
            }
        } catch (error) {
            console.log("Batch file info not supported, falling back to individual requests");
        }
        
        // Fallback: Load file info individually (but all at once)
        const infoPromises = node.fileList.map(async (filename) => {
            try {
                const response = await api.fetchApi("/shot_history/get_file_info", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: node.currentPath, filename: filename })
                });
                
                if (response.ok) {
                    const info = await response.json();
                    return { filename, info };
                }
            } catch (error) {
                console.error(`Error loading file info for ${filename}:`, error);
            }
            return null;
        });
        
        const results = await Promise.all(infoPromises);
        
        // Populate file info cache
        results.forEach(result => {
            if (result) {
                node.fileInfos[result.filename] = result.info;
            }
        });
        
        console.log(`Loaded individual file info for ${Object.keys(node.fileInfos).length} files`);
    }

    async function loadThumbnails() {
        node.thumbnails = {};
        
        for (const filename of node.filteredFileList) {
            try {
                const response = await api.fetchApi("/shot_history/get_thumbnail", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: node.currentPath, filename: filename })
                });
                
                if (response.ok) {
                    const blob = await response.blob();
                    node.thumbnails[filename] = await createImageBitmap(blob);
                }
            } catch (error) {
                console.error(`Error loading thumbnail for ${filename}:`, error);
            }
        }
        
        node.thumbnailsLoaded = true;
        node.setDirtyCanvas(true);
    }

    function applySearchFilter() {
        if (!node.searchFilter) {
            node.filteredFileList = [...node.fileList];
        } else {
            node.filteredFileList = node.fileList.filter(filename => 
                filename.toLowerCase().includes(node.searchFilter)
            );
        }
        console.log(`Filtered ${node.filteredFileList.length} files from ${node.fileList.length}`);
        applySorting();
    }

    function applySorting() {
        if (!node.sortField) return;
        
        node.filteredFileList.sort((a, b) => {
            let aVal, bVal;
            
            if (node.sortField === "filename") {
                aVal = a.toLowerCase();
                bVal = b.toLowerCase();
            } else {
                const aInfo = node.fileInfos[a];
                const bInfo = node.fileInfos[b];
                
                if (!aInfo || !bInfo) return 0;
                
                switch (node.sortField) {
                    case "size":
                        aVal = aInfo.file_size;
                        bVal = bInfo.file_size;
                        break;
                    case "width":
                        aVal = aInfo.width;
                        bVal = bInfo.width;
                        break;
                    case "height":
                        aVal = aInfo.height;
                        bVal = bInfo.height;
                        break;
                    case "created":
                        aVal = aInfo.created_timestamp;
                        bVal = bInfo.created_timestamp;
                        break;
                    case "modified":
                        aVal = aInfo.modified_timestamp;
                        bVal = bInfo.modified_timestamp;
                        break;
                    case "format":
                        aVal = aInfo.format.toLowerCase();
                        bVal = bInfo.format.toLowerCase();
                        break;
                    default:
                        return 0;
                }
            }
            
            if (aVal < bVal) return node.sortDirection === "asc" ? -1 : 1;
            if (aVal > bVal) return node.sortDirection === "asc" ? 1 : -1;
            return 0;
        });
    }

    function setSortField(field) {
        if (node.sortField === field) {
            node.sortDirection = node.sortDirection === "asc" ? "desc" : "asc";
        } else {
            node.sortField = field;
            node.sortDirection = "asc";
        }
        applySorting();
        node.setDirtyCanvas(true);
    }

    function toggleViewMode() {
        node.viewMode = node.viewMode === "grid" ? "list" : "grid";
        node.setDirtyCanvas(true);
    }

    function selectAll() {
        node.selectedFiles = [...node.filteredFileList];
        updateSelectedFilesWidget();
        node.setDirtyCanvas(true);
    }

    function clearAll() {
        node.selectedFiles = [];
        updateSelectedFilesWidget();
        node.setDirtyCanvas(true);
    }

    function toggleInfoPanel() {
        node.showInfo = !node.showInfo;
        
        if (node.showInfo && node.selectedThumbnail) {
            updateInfoPanel(node.selectedThumbnail);
        } else if (!node.showInfo && node.infoPanel) {
            node.infoPanel.style.display = 'none';
        }
        
        node.setDirtyCanvas(true);
    }

    async function loadFileInfo(filename) {
        // File info should already be loaded during refresh, but fallback to individual loading if needed
        if (node.fileInfos[filename]) {
            return node.fileInfos[filename];
        }

        // Fallback for individual file info loading (in case it wasn't loaded during batch)
        try {
            const response = await api.fetchApi("/shot_history/get_file_info", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: node.currentPath, filename: filename })
            });
            
            if (response.ok) {
                const info = await response.json();
                node.fileInfos[filename] = info;
                return info;
            }
        } catch (error) {
            console.error(`Error loading file info for ${filename}:`, error);
        }
        
        return null;
    }

    function updateSelectedFilesWidget() {
        if (selectedFilesWidget) {
            selectedFilesWidget.value = JSON.stringify(node.selectedFiles);
            console.log(`Selected files updated: ${node.selectedFiles.length} files selected`);
            console.log("Selected files:", node.selectedFiles);
        }
    }

    function getTotalContentHeight() {
        if (node.viewMode === "list") {
            const itemHeight = 40; // Height for each list item
            const headerHeight = 28; // Height for column headers
            return headerHeight + (node.filteredFileList.length * itemHeight);
        } else {
            const layout = calculateLayout();
            const { thumbnailsPerRow } = layout;
            const thumbnailHeight = THUMBNAIL_SIZE + THUMBNAIL_PADDING + 20;
            const rows = Math.ceil(node.filteredFileList.length / thumbnailsPerRow);
            return rows * thumbnailHeight;
        }
    }

    function getVisibleHeight() {
        return node.size[1] - TOP_PADDING - BOTTOM_PADDING;
    }

    function getMaxScrollOffset() {
        const totalHeight = getTotalContentHeight();
        const visibleHeight = getVisibleHeight();
        return Math.max(0, totalHeight - visibleHeight);
    }

    function clampScrollOffset() {
        const maxScroll = getMaxScrollOffset();
        node.scrollOffset = Math.max(0, Math.min(maxScroll, node.scrollOffset));
    }

    function drawScrollbar(ctx) {
        const totalHeight = getTotalContentHeight();
        const visibleHeight = getVisibleHeight();
        const maxScroll = getMaxScrollOffset();
        
        if (maxScroll <= 0) return; // No scrolling needed
        
        const scrollbarX = node.size[0] - node.scrollbarWidth;
        const scrollbarY = TOP_PADDING;
        const scrollbarHeight = visibleHeight;
        
        // Draw scrollbar track
        ctx.fillStyle = "#333";
        ctx.fillRect(scrollbarX, scrollbarY, node.scrollbarWidth, scrollbarHeight);
        
        // Calculate thumb size and position
        const thumbHeight = Math.max(20, (visibleHeight / totalHeight) * scrollbarHeight);
        const thumbY = scrollbarY + (node.scrollOffset / maxScroll) * (scrollbarHeight - thumbHeight);
        
        // Draw scrollbar thumb
        ctx.fillStyle = node.isDraggingScrollbar ? "#666" : "#555";
        ctx.fillRect(scrollbarX + 1, thumbY, node.scrollbarWidth - 2, thumbHeight);
    }

    function openFileDialog() {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = true;
        
        input.addEventListener("change", async (e) => {
            const files = Array.from(e.target.files);
            const targetDir = pathWidget ? pathWidget.value : "./shots";
            
            if (files.length === 0) return;
            
            try {
                const formData = new FormData();
                formData.append('target_dir', targetDir);
                
                for (const file of files) {
                    formData.append('file', file);
                }
                
                const response = await api.fetchApi("/shot_history/upload_file", {
                    method: "POST",
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    console.log(`Successfully uploaded ${result.files.length} files`);
                    // Force refresh after upload
                    setTimeout(() => updateFileList(true), 500);
                } else {
                    console.error("Upload error:", result.error);
                    alert(`Error uploading files: ${result.error}`);
                }
                
            } catch (error) {
                console.error("Error uploading files:", error);
                alert(`Error uploading files: ${error.message}`);
            }
        });
        
        input.click();
    }

    function getThumbnailAtPosition(x, y) {
        // Check if click is in the thumbnail area (below buttons)
        if (y < TOP_PADDING) return null;
        
        if (node.viewMode === "list") {
            // List view click detection
            const itemHeight = 40;
            const headerHeight = 28;
            const relativeY = y - TOP_PADDING - headerHeight + node.scrollOffset;
            
            // Don't allow clicks in header area
            if (y <= TOP_PADDING + headerHeight) return null;
            
            const index = Math.floor(relativeY / itemHeight);
            
            if (index >= 0 && index < node.filteredFileList.length) {
                return {
                    index: index,
                    filename: node.filteredFileList[index],
                    x: x,
                    y: y
                };
            }
        } else {
            // Grid view click detection
            const layout = calculateLayout();
            const { thumbnailsPerRow } = layout;
            
            // Calculate relative position within the scrollable content
            const relativeY = y - TOP_PADDING + node.scrollOffset;
            const relativeX = x - GRID_PADDING;
            
            if (relativeX < 0 || relativeX > node.size[0] - GRID_PADDING * 2 - node.scrollbarWidth) return null;
            
            const thumbnailWidth = THUMBNAIL_SIZE + THUMBNAIL_PADDING;
            const thumbnailHeight = THUMBNAIL_SIZE + THUMBNAIL_PADDING + 20; // +20 for filename
            
            const col = Math.floor(relativeX / thumbnailWidth);
            const row = Math.floor(relativeY / thumbnailHeight);
            
            if (col >= thumbnailsPerRow || row < 0) return null;
            
            const index = row * thumbnailsPerRow + col;
            
            if (index >= 0 && index < node.filteredFileList.length) {
                return {
                    index: index,
                    filename: node.filteredFileList[index],
                    x: col * thumbnailWidth + GRID_PADDING,
                    y: row * thumbnailHeight + TOP_PADDING - node.scrollOffset
                };
            }
        }
        
        return null;
    }

    function drawRoundedRect(ctx, x, y, width, height, radius, color) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();
    }

    // Override onDrawBackground to render thumbnails
    node.onDrawBackground = function(ctx) {
        if (this.flags.collapsed) return;
        
        // Recalculate layout for current node size
        const layout = calculateLayout();
        const { thumbnailsPerRow } = layout;
        
        // Ensure scroll offset is within bounds
        clampScrollOffset();
        
        // Draw toolbar
        drawToolbar(ctx);
        
        // Fill background
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, TOP_PADDING, this.size[0], this.size[1] - TOP_PADDING);
        
        // Note: Info panel is now a DOM element, not drawn on canvas
        
        if (!this.thumbnailsLoaded || this.filteredFileList.length === 0) {
            // Show loading or empty message
            ctx.fillStyle = COLORS.text;
            ctx.font = "14px Arial";
            ctx.textAlign = "center";
            const message = this.filteredFileList.length === 0 ? (this.searchFilter ? "No matches found" : "No images found") : "Loading thumbnails...";
            ctx.fillText(message, this.size[0] / 2, TOP_PADDING + 50);
            ctx.textAlign = "left";
            return;
        }
        
        // Draw based on view mode
        if (this.viewMode === "list") {
            drawListView(ctx, thumbnailsPerRow);
        } else {
            drawGridView(ctx, thumbnailsPerRow);
        }
        
        // Draw selection count
        if (this.selectedFiles.length > 0) {
            ctx.fillStyle = COLORS.thumbnailSelected;
            ctx.font = "12px Arial";
            ctx.fillText(`${this.selectedFiles.length} selected`, 10, TOP_PADDING - 10);
        }
        
        // Draw scrollbar
        drawScrollbar(ctx);
    };

    function drawGridView(ctx, thumbnailsPerRow) {
        const thumbnailWidth = THUMBNAIL_SIZE + THUMBNAIL_PADDING;
        const thumbnailHeight = THUMBNAIL_SIZE + THUMBNAIL_PADDING + 20;
        
        for (let i = 0; i < node.filteredFileList.length; i++) {
            const filename = node.filteredFileList[i];
            const row = Math.floor(i / thumbnailsPerRow);
            const col = i % thumbnailsPerRow;
            
            const x = col * thumbnailWidth + GRID_PADDING;
            const y = row * thumbnailHeight + TOP_PADDING - node.scrollOffset;
            
            // Skip if thumbnail is not visible (ensure they don't go above TOP_PADDING)
            if (y + thumbnailHeight < TOP_PADDING || y > node.size[1] || y < TOP_PADDING) {
                continue;
            }
            
            // Make sure thumbnail doesn't overflow the node width
            if (x + THUMBNAIL_SIZE > node.size[0] - GRID_PADDING - node.scrollbarWidth) {
                continue;
            }
            
            const isSelected = node.selectedFiles.includes(filename);
            const isHovered = node.hoveredThumbnail === filename;
            const selectionIndex = node.selectedFiles.indexOf(filename);
            
            // Draw thumbnail background
            let bgColor = COLORS.thumbnail;
            if (isSelected) {
                bgColor = COLORS.thumbnailSelected;
            } else if (isHovered) {
                bgColor = COLORS.thumbnailHover;
            }
            
            drawRoundedRect(ctx, x, y, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 4, bgColor);
            
            // Draw thumbnail image
            if (node.thumbnails[filename]) {
                ctx.save();
                ctx.beginPath();
                ctx.roundRect(x + 2, y + 2, THUMBNAIL_SIZE - 4, THUMBNAIL_SIZE - 4, 2);
                ctx.clip();
                ctx.drawImage(node.thumbnails[filename], x + 2, y + 2, THUMBNAIL_SIZE - 4, THUMBNAIL_SIZE - 4);
                ctx.restore();
            }
            
            // Draw selection indicator with order number
            if (isSelected) {
                ctx.fillStyle = "white";
                ctx.beginPath();
                ctx.arc(x + THUMBNAIL_SIZE - 10, y + 10, 10, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.fillStyle = COLORS.thumbnailSelected;
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                ctx.fillText((selectionIndex + 1).toString(), x + THUMBNAIL_SIZE - 10, y + 15);
                ctx.textAlign = "left";
            }
            
            // Draw filename
            ctx.fillStyle = COLORS.text;
            ctx.font = "10px Arial";
            const textY = y + THUMBNAIL_SIZE + 12;
            const maxWidth = THUMBNAIL_SIZE;
            
            // Truncate filename if too long
            let displayName = filename;
            const textWidth = ctx.measureText(displayName).width;
            if (textWidth > maxWidth) {
                while (ctx.measureText(displayName + "...").width > maxWidth && displayName.length > 0) {
                    displayName = displayName.slice(0, -1);
                }
                displayName += "...";
            }
            
            ctx.fillText(displayName, x, textY);
        }
    }

    function drawListView(ctx, thumbnailsPerRow) {
        const itemHeight = 40;
        const iconSize = 32;
        const headerHeight = 28;
        
        // Draw column headers
        drawListHeaders(ctx, headerHeight);
        
        const listStartY = TOP_PADDING + headerHeight;
        const adjustedScrollOffset = Math.max(0, node.scrollOffset);
        const startIdx = Math.floor(adjustedScrollOffset / itemHeight);
        const visibleHeight = getVisibleHeight() - headerHeight;
        const endIdx = Math.min(startIdx + Math.ceil(visibleHeight / itemHeight) + 2, node.filteredFileList.length);
        
        for (let i = startIdx; i < endIdx; i++) {
            const filename = node.filteredFileList[i];
            const y = listStartY + (i * itemHeight) - adjustedScrollOffset;
            
            // Skip if item is not visible
            if (y + itemHeight < listStartY || y > node.size[1] - BOTTOM_PADDING) {
                continue;
            }
            
            const isSelected = node.selectedFiles.includes(filename);
            const isHovered = node.hoveredThumbnail === filename;
            const selectionIndex = node.selectedFiles.indexOf(filename);
            
            // Draw row background
            let bgColor = i % 2 === 0 ? COLORS.background : "#252525";
            if (isSelected) {
                bgColor = COLORS.thumbnailSelected;
            } else if (isHovered) {
                bgColor = COLORS.thumbnailHover;
            }
            
            ctx.fillStyle = bgColor;
            ctx.fillRect(GRID_PADDING, y, node.size[0] - GRID_PADDING * 2 - node.scrollbarWidth, itemHeight);
            
            // Define column positions and widths (match header calculations)
            const availableWidth = node.size[0] - GRID_PADDING * 2 - node.scrollbarWidth;
            const selectionWidth = 20;
            const iconSize = 32;
            const iconPadding = 8;
            const sizeColWidth = 80;
            const dimColWidth = 90;
            const formatColWidth = 70;
            const dateColWidth = 100;
            
            // Calculate name column as remaining space (same as header)
            const fixedColumnsWidth = sizeColWidth + dimColWidth + formatColWidth + dateColWidth;
            const nameColWidth = Math.max(150, availableWidth - fixedColumnsWidth - selectionWidth - iconSize - iconPadding);
            
            // Reserve space for selection number (fixed width)
            const thumbnailX = GRID_PADDING + selectionWidth + 4;
            
            // Draw selection number in dedicated space
            if (isSelected) {
                const selectionCenterX = GRID_PADDING + selectionWidth / 2;
                ctx.fillStyle = "white";
                ctx.beginPath();
                ctx.arc(selectionCenterX, y + 20, 8, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.fillStyle = COLORS.thumbnailSelected;
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.fillText((selectionIndex + 1).toString(), selectionCenterX, y + 24);
                ctx.textAlign = "left";
            }
            
            // Draw thumbnail icon (now offset by selection width)
            if (node.thumbnails[filename]) {
                ctx.drawImage(node.thumbnails[filename], thumbnailX, y + 4, iconSize, iconSize);
            } else {
                ctx.fillStyle = COLORS.thumbnail;
                ctx.fillRect(thumbnailX, y + 4, iconSize, iconSize);
            }
            
            // Column positions - align with headers
            const nameColStart = GRID_PADDING + selectionWidth + iconSize + iconPadding; // Same as header calculation
            const nameColEnd = nameColStart + nameColWidth;
            let colX = nameColStart; // Start text at same position as header
            
            // Draw filename (with text wrapping)
            ctx.fillStyle = COLORS.text;
            ctx.font = "11px Arial";
            // Calculate the actual available width within the node bounds
            const nodeRightEdge = node.size[0] - GRID_PADDING - node.scrollbarWidth;
            const availableTextWidth = Math.min(nameColWidth - 8, nodeRightEdge - colX - 8);
            const maxNameWidth = Math.max(50, availableTextWidth); // Minimum 50px width
            const truncatedName = truncateText(ctx, filename, maxNameWidth);
            
            // Clip text to ensure it doesn't go beyond node boundaries
            ctx.save();
            ctx.beginPath();
            ctx.rect(colX, y, Math.min(nameColWidth, nodeRightEdge - colX), itemHeight);
            ctx.clip();
            ctx.fillText(truncatedName, colX, y + 16);
            ctx.restore();
            
            colX = nameColEnd; // Jump to end of name column
            
            // Draw file info columns if available
            if (node.fileInfos[filename]) {
                const info = node.fileInfos[filename];
                ctx.font = "10px Arial";
                ctx.fillStyle = "#bbb";
                
                // Size column
                ctx.fillText(info.file_size_str, colX, y + 16);
                colX += sizeColWidth;
                
                // Dimensions column
                ctx.fillText(`${info.width}×${info.height}`, colX, y + 16);
                colX += dimColWidth;
                
                // Format column
                ctx.fillText(info.format, colX, y + 16);
                colX += formatColWidth;
                
                // Date column (just the date part)
                const shortDate = info.modified_date.split(' ')[0];
                ctx.fillText(shortDate, colX, y + 16);
                
                // Second line with more details
                ctx.fillStyle = "#888";
                ctx.font = "9px Arial";
                const detailText = `${info.megapixels}MP • ${info.mode} • Created: ${info.created_date.split(' ')[0]}`;
                
                // Clip detail text to ensure it doesn't go beyond node boundaries
                ctx.save();
                const nodeRightEdge = node.size[0] - GRID_PADDING - node.scrollbarWidth;
                ctx.beginPath();
                ctx.rect(nameColStart, y + 20, nodeRightEdge - nameColStart, 15);
                ctx.clip();
                ctx.fillText(detailText, nameColStart, y + 30);
                ctx.restore();
            }
        }
    }

    function drawListHeaders(ctx, headerHeight) {
        const y = TOP_PADDING;
        const availableWidth = node.size[0] - GRID_PADDING * 2 - node.scrollbarWidth;
        
        // Header background
        ctx.fillStyle = "#333";
        ctx.fillRect(GRID_PADDING, y, availableWidth, headerHeight);
        
        // Header border
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.strokeRect(GRID_PADDING, y, availableWidth, headerHeight);
        
        // Column definitions with positions
        const selectionWidth = 20;
        const iconSize = 32;
        const iconPadding = 8;
        const sizeColWidth = 80;
        const dimColWidth = 90;
        const formatColWidth = 70;
        const dateColWidth = 100;
        
        // Calculate name column as remaining space
        const fixedColumnsWidth = sizeColWidth + dimColWidth + formatColWidth + dateColWidth;
        const nameColWidth = Math.max(150, availableWidth - fixedColumnsWidth - selectionWidth - iconSize - iconPadding);
        
        let colX = GRID_PADDING + selectionWidth + iconSize + iconPadding; // Account for selection + icon + padding
        
        const columns = [
            { text: "Name", x: colX, field: "filename", width: nameColWidth },
            { text: "Size", x: colX += nameColWidth, field: "size", width: sizeColWidth },
            { text: "Dimensions", x: colX += sizeColWidth, field: "width", width: dimColWidth },
            { text: "Format", x: colX += dimColWidth, field: "format", width: formatColWidth },
            { text: "Modified", x: colX += formatColWidth, field: "modified", width: dateColWidth },
        ];
        
        ctx.fillStyle = COLORS.text;
        ctx.font = "bold 11px Arial";
        
        columns.forEach(col => {
            const sortIndicator = node.sortField === col.field ? 
                (node.sortDirection === "asc" ? " ↑" : " ↓") : "";
            ctx.fillText(col.text + sortIndicator, col.x, y + headerHeight - 8);
        });
        
        // Store column info for click detection
        node.listColumns = columns;
        node.listHeaderHeight = headerHeight;
    }

    function truncateText(ctx, text, maxWidth) {
        const metrics = ctx.measureText(text);
        if (metrics.width <= maxWidth) {
            return text;
        }
        
        let truncated = text;
        while (ctx.measureText(truncated + "...").width > maxWidth && truncated.length > 0) {
            truncated = truncated.slice(0, -1);
        }
        return truncated + "...";
    }

    // Create or update the DOM-based info panel
    function createInfoPanel() {
        if (node.infoPanel) {
            node.infoPanel.parentNode.removeChild(node.infoPanel);
        }
        
        node.infoPanel = document.createElement('div');
        node.infoPanel.style.position = 'absolute';
        node.infoPanel.style.background = 'rgba(42, 42, 42, 0.95)';
        node.infoPanel.style.border = '1px solid #666';
        node.infoPanel.style.borderRadius = '6px';
        node.infoPanel.style.padding = '12px';
        node.infoPanel.style.fontSize = '11px';
        node.infoPanel.style.fontFamily = 'Monaco, "Lucida Console", monospace';
        node.infoPanel.style.color = '#fff';
        node.infoPanel.style.lineHeight = '1.4';
        node.infoPanel.style.maxWidth = '300px';
        node.infoPanel.style.minWidth = '250px';
        node.infoPanel.style.zIndex = '10000';
        node.infoPanel.style.wordWrap = 'break-word';
        node.infoPanel.style.userSelect = 'text';
        node.infoPanel.style.cursor = 'text';
        node.infoPanel.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
        node.infoPanel.style.backdropFilter = 'blur(8px)';
        
        document.body.appendChild(node.infoPanel);
    }

    function updateInfoPanel(filename) {
        if (!node.showInfo) {
            if (node.infoPanel) {
                node.infoPanel.style.display = 'none';
            }
            return;
        }
        
        if (!node.infoPanel) {
            createInfoPanel();
        }
        
        node.infoPanel.style.display = 'block';
        
        const info = node.fileInfos[filename];
        if (!info) {
            node.infoPanel.innerHTML = '<div style="color: #999;">Loading file information...</div>';
            return;
        }
        
        const content = `
<div style="color: #4a90e2; font-weight: bold; margin-bottom: 8px;">📄 ${info.filename}</div>
<div><strong>Format:</strong> ${info.format}</div>
<div><strong>Dimensions:</strong> ${info.width} × ${info.height} px (${info.megapixels} MP)</div>
<div><strong>Aspect Ratio:</strong> ${info.aspect_ratio}:1</div>
<div><strong>File Size:</strong> ${info.file_size_str}</div>
<div><strong>Color Mode:</strong> ${info.mode}</div>
<div><strong>DPI:</strong> ${info.dpi}</div>
<div><strong>Created:</strong> ${info.created_date}</div>
<div><strong>Modified:</strong> ${info.modified_date}</div>
<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #555; color: #999; font-size: 9px;">
Path: ${info.full_path}
</div>`;
        
        node.infoPanel.innerHTML = content;
        positionInfoPanel();
    }

    function positionInfoPanel() {
        if (!node.infoPanel || !node.showInfo) return;
        
        // Try to get canvas safely
        let canvas = null;
        let canvasRect = null;
        
        try {
            if (node.graph && node.graph.canvas && node.graph.canvas.canvas) {
                canvas = node.graph.canvas.canvas;
                canvasRect = canvas.getBoundingClientRect();
            } else {
                // Fallback: find canvas in DOM
                canvas = document.querySelector('canvas');
                if (canvas) {
                    canvasRect = canvas.getBoundingClientRect();
                }
            }
        } catch (e) {
            console.warn("Could not get canvas for info panel positioning:", e);
        }
        
        // If we can't get canvas positioning, use a simple fallback
        if (!canvasRect) {
            node.infoPanel.style.left = '50px';
            node.infoPanel.style.top = '100px';
            return;
        }
        
        const nodePos = node.pos || [0, 0];
        const nodeSize = node.size || [400, 300];
        
        // Get scale and offset safely
        let scale = 1;
        let offset = [0, 0];
        
        try {
            if (node.graph && node.graph.canvas && node.graph.canvas.ds) {
                scale = node.graph.canvas.ds.scale || 1;
                offset = node.graph.canvas.ds.offset || [0, 0];
            }
        } catch (e) {
            console.warn("Could not get canvas transform:", e);
        }
        
        // Calculate position relative to viewport
        const left = canvasRect.left + (nodePos[0] + nodeSize[0] + 10) * scale + offset[0];
        const top = canvasRect.top + nodePos[1] * scale + offset[1];
        
        // Keep panel within viewport
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const panelWidth = parseInt(node.infoPanel.style.minWidth) || 250;
        
        let finalLeft = left;
        let finalTop = top;
        
        // Adjust horizontal position if panel would go off-screen
        if (finalLeft + panelWidth > viewportWidth) {
            finalLeft = Math.max(10, canvasRect.left + (nodePos[0] - panelWidth - 10) * scale + offset[0]);
        }
        
        // Adjust vertical position if needed
        if (finalTop + 200 > viewportHeight) {
            finalTop = Math.max(10, viewportHeight - 220);
        }
        
        // Ensure panel stays within viewport bounds
        finalLeft = Math.max(10, Math.min(finalLeft, viewportWidth - panelWidth - 10));
        finalTop = Math.max(10, Math.min(finalTop, viewportHeight - 200));
        
        node.infoPanel.style.left = finalLeft + 'px';
        node.infoPanel.style.top = finalTop + 'px';
    }

    // Handle mouse interactions
    node.onMouseDown = function(event) {
        const localX = event.canvasX - this.pos[0];
        const localY = event.canvasY - this.pos[1];
        
        // Check if clicking on toolbar
        if (this.toolbarIcons && localY >= this.toolbarY && localY <= this.toolbarY + this.toolbarHeight) {
            const iconIndex = getClickedToolbarIcon(localX);
            if (iconIndex !== -1) {
                handleToolbarClick(iconIndex);
                return true;
            }
        }
        
        // Check if clicking on list view headers
        if (this.viewMode === "list" && this.listColumns && localY >= TOP_PADDING && localY <= TOP_PADDING + this.listHeaderHeight) {
            // Find which column was clicked
            for (const col of this.listColumns) {
                if (localX >= col.x - 10 && localX <= col.x + col.width) {
                    console.log(`Clicked on column: ${col.field}`);
                    setSortField(col.field);
                    return true;
                }
            }
        }
        
        // Check if clicking on scrollbar
        const scrollbarX = this.size[0] - this.scrollbarWidth;
        if (localX >= scrollbarX && localY >= TOP_PADDING && localY <= this.size[1] - BOTTOM_PADDING) {
            const maxScroll = getMaxScrollOffset();
            if (maxScroll > 0) {
                this.isDraggingScrollbar = true;
                this.scrollStartY = localY;
                this.scrollStartOffset = this.scrollOffset;
                return true;
            }
        }
        
        const thumbnail = getThumbnailAtPosition(localX, localY);
        if (thumbnail) {
            // Set selected thumbnail for info display
            this.selectedThumbnail = thumbnail.filename;
            
            // Update DOM info panel (file info should already be loaded)
            updateInfoPanel(thumbnail.filename);
            
            // Load file info if not already loaded (fallback)
            if (!this.fileInfos[thumbnail.filename]) {
                loadFileInfo(thumbnail.filename).then(() => {
                    updateInfoPanel(thumbnail.filename);
                    this.setDirtyCanvas(true);
                });
            }
            
            // Toggle selection with ordering
            const index = this.selectedFiles.indexOf(thumbnail.filename);
            if (index !== -1) {
                // Remove from selection
                this.selectedFiles.splice(index, 1);
            } else {
                // Add to selection
                this.selectedFiles.push(thumbnail.filename);
            }
            
            updateSelectedFilesWidget();
            this.setDirtyCanvas(true);
            return true;
        }
        
        return false;
    };

    node.onMouseMove = function(event) {
        const localX = event.canvasX - this.pos[0];
        const localY = event.canvasY - this.pos[1];
        
        // Handle scrollbar dragging
        if (this.isDraggingScrollbar) {
            const maxScroll = getMaxScrollOffset();
            const visibleHeight = getVisibleHeight();
            const deltaY = localY - this.scrollStartY;
            const scrollRatio = deltaY / visibleHeight;
            this.scrollOffset = this.scrollStartOffset + scrollRatio * maxScroll;
            clampScrollOffset(); // Ensure scrolling stays within bounds
            this.setDirtyCanvas(true);
            return true;
        }
        
        // Check toolbar hover
        let toolbarHover = -1;
        if (this.toolbarIcons && localY >= this.toolbarY && localY <= this.toolbarY + this.toolbarHeight) {
            toolbarHover = getClickedToolbarIcon(localX);
        }
        
        if (toolbarHover !== this.hoveredToolbarIcon) {
            this.hoveredToolbarIcon = toolbarHover;
            this.setDirtyCanvas(true);
        }
        
        const thumbnail = getThumbnailAtPosition(localX, localY);
        const newHovered = thumbnail ? thumbnail.filename : null;
        
        if (newHovered !== this.hoveredThumbnail) {
            this.hoveredThumbnail = newHovered;
            
            // Update info panel position when hovering changes
            if (this.showInfo && newHovered) {
                this.selectedThumbnail = newHovered;
                updateInfoPanel(newHovered);
                
                // Load file info if not already loaded (fallback - should rarely happen)
                if (!this.fileInfos[newHovered]) {
                    loadFileInfo(newHovered).then(() => {
                        updateInfoPanel(newHovered);
                    });
                }
            }
            
            this.setDirtyCanvas(true);
        }
        
        // Update info panel position when node moves
        if (this.showInfo && this.infoPanel) {
            positionInfoPanel();
        }
        
        return false;
    };

    node.onMouseUp = function(event) {
        if (this.isDraggingScrollbar) {
            this.isDraggingScrollbar = false;
            this.setDirtyCanvas(true);
            return true;
        }
        return false;
    };

    // Add mouse wheel scrolling
    const originalOnMouseWheel = node.onMouseWheel;
    node.onMouseWheel = function(event) {
        if (originalOnMouseWheel) {
            originalOnMouseWheel.apply(this, arguments);
        }
        
        const maxScroll = getMaxScrollOffset();
        if (maxScroll > 0) {
            const scrollSpeed = 40;
            this.scrollOffset += event.deltaY * scrollSpeed;
            clampScrollOffset(); // Ensure scrolling stays within bounds
            this.setDirtyCanvas(true);
            return true;
        }
        
        return false;
    };

    // Watch for path changes
    if (pathWidget) {
        const originalCallback = pathWidget.callback;
        pathWidget.callback = () => {
            if (originalCallback) {
                originalCallback.apply(pathWidget, arguments);
            }
            
            console.log("Path changed to:", pathWidget.value);
            
            // Clear selection when path changes
            node.selectedFiles = [];
            updateSelectedFilesWidget();
            
            // Force update file list for path change
            setTimeout(() => updateFileList(true), 100);
        };
    }

    // Handle node resize
    const originalOnResize = node.onResize;
    node.onResize = function() {
        if (originalOnResize) {
            originalOnResize.apply(this, arguments);
        }
        // Trigger redraw on resize to update layout
        this.setDirtyCanvas(true);
        
        // Update info panel position on resize
        if (this.showInfo && this.infoPanel) {
            positionInfoPanel();
        }
    };

    // Handle node removal - cleanup DOM elements
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function() {
        if (originalOnRemoved) {
            originalOnRemoved.call(this);
        }
        // Cleanup info panel
        if (this.infoPanel && this.infoPanel.parentNode) {
            this.infoPanel.parentNode.removeChild(this.infoPanel);
            this.infoPanel = null;
        }
    };

    // Initial load
    setTimeout(() => updateFileList(true), 100);
} 